import os, sys, time, random
import argparse
from pathlib import Path

IMAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = IMAGE_ROOT.parent
for path in (str(IMAGE_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torchvision.models import resnet50
from utils import *
from data import load_cifar10, load_cifar100, load_svhn, ImageNetDataset
from models.wide_resnet import Wide_ResNet
from models.densenet import densenet121
from models.resnet import resnet3_96

from optim.AMUSE import AMUSE
from optim.muon import MuonWithAuxSGD
from optim.schedulefree import SGDScheduleFree
import itertools
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import wandb
########################################################################################################################
#  Training Baseline
########################################################################################################################

parser = argparse.ArgumentParser(description='Trains ResNet on CIFAR', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default='./data', help='Path to dataset')
parser.add_argument('--dataset', type=str, default='cifar10',choices=['cifar10', 'cifar100', 'svhn', 'imagenet'])
parser.add_argument('--arch', type=str, default='wideresnet', choices=['wideresnet', 'resnet50', 'resnet3-96', 'densenet'])

# Optimization options
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=128, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=10, help='Base Learning Rate.')
parser.add_argument('--sgd_learning_rate', type=float, default=10, help='Learning rate for Muon/AMUSE non-Muon parameters.')
parser.add_argument('--optimizer', type=str, default='amuse', help='optimizer', choices=['sgd', 'muon', 'sf-sgd', 'amuse'])
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--beta', type=float, default=0.9, help='Schedulefree Momentum.')
parser.add_argument('--amuse_aux_opt', type=str, default='sgd', choices=['sgd', 'adamw'], help='AMUSE update used for non-Muon parameters and Muon scaling.')
parser.add_argument('--scheduler', type=str, default=None, choices=['cos', 'none', 'warmupconst'], help='Learning rate scheduler for SGD/Muon. Defaults to cos for SGD/Muon and none for schedule-free optimizers.')
parser.add_argument('--warmup_ratio', type=float, default=0.05, help='The ratio of warmup steps to total steps.')
parser.add_argument('--rho', type=float, default=1.0, help='AMUSE window rho parameter.')
parser.add_argument('--decay', type=float, default=0.0001, help='Weight decay (L2 penalty).')
parser.add_argument("--weight_decay_at_y", type=float, default=0, help="SF:Weight decay calculated at the y point. (default 0)")
parser.add_argument("--weight_lr_power", type=float, default=2, help="SF:During warmup, the weights in the average will be equal to lr raised to this power. Set to 0 for no weighting (default 2.0).")
parser.add_argument("--r", type=float, default=0, help="SF:Use polynomial weighting in the average with power r (default 0.0)")
parser.add_argument('--ema', action='store_true', default=False, help='Track an EMA copy of model parameters and evaluate it.')
parser.add_argument('--emacoef', type=float, nargs='+', default=[0.999], help='EMA coefficient(s) for model parameters.')
parser.add_argument('--ema_bn_batches', type=int, default=50, help='Number of train batches used to refresh EMA BatchNorm statistics before evaluation. Set 0 to disable.')
parser.add_argument('--wandb', action='store_true', default=False, help='Enable Weights & Biases logging.')
parser.add_argument('--wandb_project', type=str, default=None, help='Weights & Biases project name.')
parser.add_argument('--wandb_entity', type=str, default=None, help='Weights & Biases entity name.')
# Checkpoints and Dynamics
parser.add_argument('--print_freq', default=200, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='./save', help='Deprecated; local logs/figures are not saved.')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',default= False, help='evaluate model on validation set')
# Acceleration
parser.add_argument('--gpu', type=str, default=0)
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, default='42', help='manual seed')

args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()
args.rank = int(os.environ.get("RANK", 0))
args.world_size = int(os.environ.get("WORLD_SIZE", 1))
args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
args.distributed = args.dataset == 'imagenet' and args.world_size > 1 and args.use_cuda

if args.dataset == 'imagenet' and args.use_cuda:
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        args.device = f'cuda:{args.local_rank}'
    else:
        torch.cuda.set_device(int(args.gpu))
        args.device = f'cuda:{args.gpu}'
else:
    if args.use_cuda:
        torch.cuda.set_device(int(args.gpu))
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'
args.is_main_process = (not args.distributed) or args.rank == 0

for ema_coef in args.emacoef:
    if not 0.0 <= ema_coef < 1.0:
        raise ValueError("--emacoef values must be in [0, 1).")
if len(set(args.emacoef)) != len(args.emacoef):
    raise ValueError("--emacoef values must be unique.")

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
seed = args.manualSeed + getattr(args, 'rank', 0)
random.seed(seed)
torch.manual_seed(seed)
if args.use_cuda:
    torch.cuda.manual_seed_all(seed)
cudnn.benchmark = True


def main():
    # Init logger. Keep file-like object for existing print_log calls, but do not
    # create local log directories or files.
    log = open(os.devnull, 'w')
    
    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.num_samples = 50000
        args.num_iter = args.num_samples/args.batch_size
        train_loader, test_loader = load_cifar10(args)
    if args.dataset == 'cifar100':
        args.num_classes = 100
        args.num_samples = 50000
        args.num_iter = args.num_samples/args.batch_size
        train_loader, test_loader = load_cifar100(args)
    if args.dataset == 'svhn':
        args.num_classes = 10
        args.num_samples = 73257
        args.num_iter = args.num_samples/args.batch_size
        train_loader, test_loader = load_svhn(args)
    if args.dataset == 'imagenet':
        args.num_classes = 1000
        trainset = ImageNetDataset.get_ImageNet_train(os.path.join(args.data_path, 'train'))
        testset = ImageNetDataset.get_ImageNet_test(os.path.join(args.data_path, 'val'))

        args.num_samples = len(trainset)
        effective_world_size = args.world_size if args.distributed else 1
        args.num_iter = args.num_samples / (args.batch_size * effective_world_size)

        train_sampler = DistributedSampler(trainset, num_replicas=effective_world_size, rank=args.rank, shuffle=True) if args.distributed else None
        val_sampler = DistributedSampler(testset, num_replicas=effective_world_size, rank=args.rank, shuffle=False) if args.distributed else None

        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, pin_memory=True, num_workers=args.workers, drop_last=False)
        test_loader = DataLoader(testset, batch_size=args.batch_size * 2, shuffle=False, sampler=val_sampler, pin_memory=True, num_workers=args.workers)

    # Init model, criterion, and optimizer
    print_log("=> creating model '{}'".format(args.arch), log, args)
    if args.arch == 'wideresnet':
        net = Wide_ResNet(num_classes=args.num_classes)
    elif args.arch == 'resnet50':
        net = resnet50(num_classes=args.num_classes)
    elif args.arch == 'densenet':
        net = densenet121(num_classes=args.num_classes)
    elif args.arch == 'resnet3-96':
        net = resnet3_96(num_classes=args.num_classes)
    project_name = args.wandb_project or f"AMUSE-{args.dataset}-{args.arch}"
    print_log("=> network :\n {}".format(net), log, args)

    # Set run name
    if args.optimizer not in ['sgd', 'muon'] and args.scheduler is not None and args.scheduler != 'none':
        raise ValueError(f"--scheduler {args.scheduler} is only supported for sgd and muon optimizers.")
    effective_scheduler = (args.scheduler or 'cos') if args.optimizer in ['sgd', 'muon'] else 'none'
    run_name = f"{args.optimizer}-lr{args.learning_rate}-beta{args.momentum}-decay{args.decay}-scheduler{effective_scheduler}-warmup{args.warmup_ratio}"
    if args.optimizer == 'muon':
        run_name = f"{args.optimizer}-lr{args.learning_rate}-sgdlr{args.sgd_learning_rate}-beta{args.momentum}-decay{args.decay}-scheduler{effective_scheduler}-warmup{args.warmup_ratio}"
    elif args.optimizer == 'amuse':
        run_name = f"{args.optimizer}-lr{args.learning_rate}-sgdlr{args.sgd_learning_rate}-beta{args.beta}-rho{args.rho}-aux{args.amuse_aux_opt}-decay{args.decay}-scheduler{effective_scheduler}-warmup{args.warmup_ratio}"
    if args.ema:
        ema_name = "-".join(f"{ema_coef:g}" for ema_coef in args.emacoef)
        run_name = f"{run_name}-ema{ema_name}"
    
    # Initialize wandb
    if args.is_main_process and args.wandb:
        wandb.init(project=project_name, entity=args.wandb_entity, name=run_name, config=args)
    else:
        wandb.init(mode='disabled')

    net = net.to(args.device)
    model = net.module if hasattr(net, "module") else net

    if args.dataset == 'imagenet' and args.distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], output_device=args.local_rank)
    elif args.dataset == 'imagenet':
        net = torch.nn.parallel.DataParallel(net).to(args.device)
    ema_models = {ema_coef: create_ema_model(net) for ema_coef in args.emacoef} if args.ema else {}

    # define loss function (criterion)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    warmup_steps = int(args.epochs * args.num_iter * args.warmup_ratio) 
    state = {k: v for k, v in args._get_kwargs()}

    # define optimizer and scheduler
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                                    weight_decay=state['decay'], nesterov=True)
        scheduler = create_scheduler(optimizer, args)

    elif args.optimizer == "muon":
        hidden_weights = [p for p in model.parameters() if p.ndim >= 2 and p is not model.fc.weight]
        nonhidden_params = [p for p in net.parameters() if p.ndim < 2]
        head_params = [model.fc.weight]
        param_groups = [
            dict(
                params=hidden_weights,
                use_muon=True,
                lr=args.learning_rate,
                weight_decay=args.decay,
                momentum=args.momentum,
            ),
            dict(
                params=nonhidden_params + head_params,
                use_muon=False,
                lr=args.sgd_learning_rate,
                momentum=args.momentum,
                weight_decay=args.decay,
            ),
        ]
        optimizer = MuonWithAuxSGD(param_groups)
        scheduler = create_scheduler(optimizer, args)

    elif args.optimizer == 'sf-sgd':
        optimizer = SGDScheduleFree(net.parameters(), 
                        lr=args.learning_rate, 
                        weight_decay=args.decay, 
                        momentum=args.beta,
                        warmup_steps=warmup_steps,
                        r=args.r,
                        weight_lr_power=args.weight_lr_power)
        optimizer.train()
        scheduler=None

    elif args.optimizer == 'amuse':
        if warmup_steps <= 0:
            raise ValueError("--warmup_ratio must create at least one warmup step for AMUSE.")
        hidden_weights = [p for p in net.parameters() if p.ndim >= 2 and p is not model.fc.weight]
        nonhidden_params = [p for p in net.parameters() if p.ndim < 2]
        head_params = [model.fc.weight]
        param_groups = [
            dict(
                params=nonhidden_params + head_params,
                use_muon=False,
                update_type=args.amuse_aux_opt,
                lr=args.sgd_learning_rate,
                weight_decay=args.decay,
            ),
            dict(
                params=hidden_weights,
                use_muon=True,
                aux_update_type=args.amuse_aux_opt,
                lr=args.learning_rate,
                weight_decay=args.decay,
                momentum=args.momentum,
            ),
        ]
        optimizer = AMUSE(
            param_groups,
            weight_decay_at_y=args.weight_decay_at_y,
            beta1=args.beta,
            warmup_steps=warmup_steps,
            r=args.r,
            rho=args.rho,
            weight_lr_power=args.weight_lr_power,
        )
        optimizer.train()
        scheduler = None

    recorder = RecorderMeter(args.epochs)
    ema_recorders = {ema_coef: RecorderMeter(args.epochs) for ema_coef in args.emacoef} if args.ema else {}
    
    # evaluation
    if args.evaluate:
        if args.optimizer in ['amuse', 'sf-sgd']:
            optimizer.eval()
            with torch.no_grad():
                for batch in itertools.islice(train_loader, 50):
                    inputs = batch[0].to(args.device)
                    model(inputs)

        time1 = time.time()
        validate(test_loader, args, net, criterion, log) #
        if args.ema:
            for ema_coef, ema_model in ema_models.items():
                label = ema_label(ema_coef)
                refresh_bn_stats(ema_model, train_loader, args, args.ema_bn_batches)
                validate(test_loader, args, ema_model, criterion, log, prefix=f'{label} Test')
        time2 = time.time()
        print('function took %0.3f ms' % ((time2 - time1) * 1000.0))
        return

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()

    for epoch in range(args.epochs):
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s}'.format(time_string(), epoch, args.epochs,
                                                                                   need_time) \
            + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                               100 - recorder.max_accuracy(False)), log, args)

        # train for one epoch
        train_acc, train_loss = train(train_loader, args, net, criterion, optimizer, scheduler, epoch, log, ema_models)

        # evaluate on validation set
        val_acc, val_loss = validate(test_loader, args, net, criterion, log, train_loader, optimizer)
        recorder.update(epoch, train_loss, train_acc, val_loss, val_acc)

        ema_metrics = {}
        if args.ema:
            for ema_coef, ema_model in ema_models.items():
                label = ema_label(ema_coef)
                refresh_bn_stats(ema_model, train_loader, args, args.ema_bn_batches)
                ema_val_acc, ema_val_loss = validate(test_loader, args, ema_model, criterion, log, prefix=f'{label} Test')
                ema_recorders[ema_coef].update(epoch, train_loss, train_acc, ema_val_loss, ema_val_acc)
                ema_metrics.update({
                    f'{label}_validation_accuracy': ema_val_acc,
                    f'{label}_validation_loss': ema_val_loss,
                    f'{label}_best_validation_accuracy': ema_recorders[ema_coef].max_accuracy(False),
                    f'{label}_coef': ema_coef,
                })

        if args.is_main_process:
            if scheduler:
                current_learning_rate = scheduler.get_last_lr()[0]
            else:
                current_learning_rate = optimizer.param_groups[0]['lr']
            
            log_dict = {
                        'epoch': epoch,
                        'train_accuracy': train_acc,
                        'train_loss': train_loss,
                        'validation_accuracy': val_acc,
                        'validation_loss': val_loss,
                        'learning_rate': current_learning_rate,
                        'best_validation_accuracy': recorder.max_accuracy(False),
                    }
            if args.optimizer == 'amuse':
                log_dict['beta'] = optimizer.beta1
            if args.ema:
                log_dict.update(ema_metrics)
            wandb.log(log_dict)

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

    log.close()
    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()

# train function (forward, backward, update)
def train(train_loader, args, model, criterion, optimizer, scheduler, epoch, log, ema_models=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    if args.optimizer in ['amuse', 'sf-sgd']:
        optimizer.train()

    if isinstance(getattr(train_loader, 'sampler', None), DistributedSampler):
        train_loader.sampler.set_epoch(epoch)

    end = time.time()
    
    for t, (input, target) in enumerate(train_loader):
        x = input.to(args.device)
        y = target.to(args.device)

        optimizer.zero_grad()
        # compute output
        output = model(x)
        loss = criterion(output, y)
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, y, topk=(1, 5))
        losses.update(loss.item(), len(y))
        top1.update(prec1.item(), len(y))
        top5.update(prec5.item(), len(y))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        if ema_models is not None:
            for ema_coef, ema_model in ema_models.items():
                update_ema_model(ema_model, model, ema_coef)
        
        if scheduler is not None:
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if t % args.print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                epoch, t, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log, args)
    print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100 - top1.avg), log, args)
    train_acc = torch.tensor([top1.avg], device=args.device)
    train_loss = torch.tensor([losses.avg], device=args.device)

    if args.distributed:
        dist.all_reduce(train_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        train_acc = (train_acc / args.world_size).item()
        train_loss = (train_loss / args.world_size).item()
    else:
        train_acc = train_acc.item()
        train_loss = train_loss.item()

    return train_acc, train_loss


def validate(test_loader, args, model, criterion, log, train_loader=None, optimizer=None, prefix='Test'): 
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.optimizer in ['amuse', 'sf-sgd'] and optimizer is not None and train_loader is not None:
        model.train()
        optimizer.eval()
        with torch.no_grad():
            for batch in itertools.islice(train_loader, 50):
                inputs = batch[0].to(args.device)
                model(inputs)

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            
            y = target.to(args.device)
            x = input.to(args.device)

            # compute output
            output = model(x)
            loss = criterion(output, y)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, y, topk=(1, 5))
            losses.update(loss.item(), len(y))
            top1.update(prec1.item(), len(y))
            top5.update(prec5.item(), len(y))

        print_log('  **{prefix}** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(prefix=prefix, top1=top1, top5=top5, error1=100 - top1.avg), log, args)

    val_acc = torch.tensor([top1.avg], device=args.device)
    val_loss = torch.tensor([losses.avg], device=args.device)

    if args.distributed:
        dist.all_reduce(val_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        val_acc = (val_acc / args.world_size).item()
        val_loss = (val_loss / args.world_size).item()
    else:
        val_acc = val_acc.item()
        val_loss = val_loss.item()
    return val_acc, val_loss


if __name__ == '__main__':
    main()
