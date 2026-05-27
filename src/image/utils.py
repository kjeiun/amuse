import time
import copy
import itertools
import numpy as np
import random
import math
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, LambdaLR
import torch

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


class RecorderMeter(object):
  """Computes and stores the minimum loss value and its epoch index"""
  def __init__(self, total_epoch):
    self.reset(total_epoch)

  def reset(self, total_epoch):
    assert total_epoch > 0
    self.total_epoch   = total_epoch
    self.current_epoch = 0
    self.epoch_losses  = np.zeros((self.total_epoch, 2), dtype=np.float64) # [epoch, train/val]
    self.epoch_losses  = self.epoch_losses - 1

    self.epoch_accuracy= np.zeros((self.total_epoch, 2), dtype=np.float64) # [epoch, train/val]
    self.epoch_accuracy= self.epoch_accuracy

  def update(self, idx, train_loss, train_acc, val_loss, val_acc):
    assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(self.total_epoch, idx)
    self.epoch_losses  [idx, 0] = train_loss
    self.epoch_losses  [idx, 1] = val_loss
    self.epoch_accuracy[idx, 0] = train_acc
    self.epoch_accuracy[idx, 1] = val_acc
    self.current_epoch = idx + 1
    return self.max_accuracy(False) == val_acc

  def max_accuracy(self, istrain):
    if self.current_epoch <= 0: return 0
    if istrain: return self.epoch_accuracy[:self.current_epoch, 0].max()
    else:       return self.epoch_accuracy[:self.current_epoch, 1].max()
    

def time_string():
  ISOTIMEFORMAT='%Y-%m-%d %X'
  string = '[{}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()+28800) ))
  return string

def convert_secs2time(epoch_time):
  need_hour = int(epoch_time / 3600)
  need_mins = int((epoch_time - 3600*need_hour) / 60)
  need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
  return need_hour, need_mins, need_secs

def time_file_str():
  ISOTIMEFORMAT='%Y-%m-%d'
  string = '{}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string + '-{}'.format(random.randint(1, 10000))

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print ('%s function took %0.3f ms' % (f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap




def warmup_cosine_scheduler(optimizer, epochs, steps_per_epoch, warmup_ratio=0.0, eta_min=0.0):

    total_steps = int(epochs * steps_per_epoch)
    warmup_steps = int(total_steps * warmup_ratio)

    if warmup_steps > 0:
        warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)

    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=eta_min)

    if warmup_steps > 0:
        sched = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    else:
        sched = cosine

    return sched


def warmup_constant_scheduler(optimizer, epochs, steps_per_epoch, warmup_ratio=0.0):

    total_steps = int(epochs * steps_per_epoch)
    warmup_steps = int(total_steps * warmup_ratio)

    if warmup_steps > 0:
        return LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)

    return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)


def create_scheduler(optimizer, args):
    scheduler_name = args.scheduler or 'cos'
    if scheduler_name == 'none':
        return None
    if scheduler_name == 'cos':
        return warmup_cosine_scheduler(optimizer, args.epochs, args.num_iter, warmup_ratio=args.warmup_ratio, eta_min=0)
    if scheduler_name == 'warmupconst':
        return warmup_constant_scheduler(optimizer, args.epochs, args.num_iter, warmup_ratio=args.warmup_ratio)
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def create_ema_model(model):
    ema_model = copy.deepcopy(unwrap_model(model))
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)
    return ema_model


def ema_label(ema_coef):
    return f"ema_{ema_coef:g}".replace('.', 'p').replace('-', 'm')


@torch.no_grad()
def update_ema_model(ema_model, model, ema_coef):
    source_model = unwrap_model(model)
    for ema_param, param in zip(ema_model.parameters(), source_model.parameters()):
        ema_param.mul_(ema_coef).add_(param.detach(), alpha=1.0 - ema_coef)
    for ema_buffer, buffer in zip(ema_model.buffers(), source_model.buffers()):
        if not torch.is_floating_point(ema_buffer):
            ema_buffer.copy_(buffer.detach())


@torch.no_grad()
def refresh_bn_stats(model, train_loader, args, num_batches):
    if num_batches <= 0:
        return

    bn_modules = [
        module for module in model.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.track_running_stats
    ]
    if not bn_modules:
        return

    was_training = model.training
    momenta = {module: module.momentum for module in bn_modules}
    for module in bn_modules:
        module.reset_running_stats()
        module.momentum = None

    model.train()
    for batch in itertools.islice(train_loader, num_batches):
        inputs = batch[0].to(args.device)
        model(inputs)

    for module, momentum in momenta.items():
        module.momentum = momentum
    model.train(was_training)


def print_log(print_string, log, args=None):
    if args is not None and not getattr(args, 'is_main_process', True):
        return
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
