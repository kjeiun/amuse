from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
import math

from .arxiv import get_arxiv_2000, get_arxiv_full
from .benchmarks import SUPPORTED_TASK_MAP, tknzr
from .c4 import get_c4_data
from .fineweb import get_fineweb_data
from .fineweb_edu import get_fineweb_edu_data
from .openwebtext2 import get_openwebtext2_data
from .redpajama import get_redpajama_data, get_redpajamav2_data
from .shakespeare import get_shakespeare_data
from .slimpajama import get_slimpajama_data
from .wikitext import get_wikitext_data



def get_dataset(args) -> Dict[str, np.ndarray]:
    """Fetch the right dataset given by the args.dataset parameter. The logic for each dataset is
    contained in its own python file. The expected format at the moment is a dictionary of np.memmap
    containing two keys: 'train' and 'val', corresponding to the tokenized training and validation data.
    """
    if args.dataset == "wikitext":
        return get_wikitext_data(args.datasets_dir)
    if args.dataset == "shakespeare-char":
        return get_shakespeare_data(args.datasets_dir)
    if args.dataset == "arxiv2000":
        return get_arxiv_2000(args.datasets_dir)
    if args.dataset == "arxiv":
        return get_arxiv_full(args.datasets_dir)
    if args.dataset == "arxiv+wiki":
        arxiv_data = get_arxiv_full(args.datasets_dir)
        wiki_data = get_wikitext_data(args.datasets_dir)
        train_data = np.concatenate((arxiv_data["train"], wiki_data["train"]))
        val_data = np.concatenate((arxiv_data["val"], wiki_data["val"]))
        return {"train": train_data, "val": val_data}
    if args.dataset == "openwebtext2":
        return get_openwebtext2_data(args.datasets_dir)
    if args.dataset == "redpajama":
        return get_redpajama_data(args.datasets_dir)
    if args.dataset == "redpajamav2":
        return get_redpajamav2_data(args.datasets_dir)
    if args.dataset == "slimpajama":
        return get_slimpajama_data(args.datasets_dir)
    if args.dataset == "fineweb":
        return get_fineweb_data(args.datasets_dir)
    if args.dataset == "finewebedu":
        return get_fineweb_edu_data(args.datasets_dir)
    if args.dataset == "c4":
        return get_c4_data(args.datasets_dir)
    if args.dataset in SUPPORTED_TASK_MAP:
        return get_benchmark_task(args.dataset)
    else:
        raise NotImplementedError(f"Unknow dataset key '{args.dataset}'")


def get_benchmark_task(name, **kwargs):
    """Fetch the right benchmark task given by the name parameter. The logic for each task is
    contained in its own python file.
    """
    try:
        fn = SUPPORTED_TASK_MAP[name]
    except KeyError:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: {sorted(SUPPORTED_TASK_MAP.keys())}"
        )
    return fn(**kwargs)


class DataReader:
    def __init__(
        self,
        data_src,
        batch_size,
        sequence_length,
        seed=1337,
        with_replacement=False,
        auto_shard=True,
        keep_in_ram=False,
    ):
        if isinstance(data_src, (str, Path)):
            self.data_path = Path(data_src)
            self.keep_in_ram = keep_in_ram
            if keep_in_ram:
                self.data = np.array(
                    np.memmap(self.data_path, dtype=np.uint16, mode="r")
                )
            else:
                self.data = None
        elif isinstance(data_src, (np.ndarray, np.memmap)):
            self.data_path = None
            self.data = data_src
            self.keep_in_ram = True

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.seed = seed
        self.with_replacement = with_replacement

        self.num_tokens = len(self._get_data())

        if auto_shard and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            print(
                f"Distributed DataReader Initialized for Worker {self.rank}/{self.world_size}"
            )
        else:
            self.world_size = 1
            self.rank = 0

        # Sampling without replacement
        self.last_epoch = None
        self.order = None
        self.epoch_offset = None
        self.step = 0
        self.num_batches_of_seqlen = 0
        if not with_replacement:
            self._shuffle_epoch(0)

    def __len__(self):
        # Length in valid start indices for a sequence
        # Extra -1 to have a valid next token for the final token of the last idx
        return self.num_tokens - self.sequence_length - 1

    def _get_data(self):
        if self.data is not None:
            return self.data
        else:
            # Construct the memmap each time to avoid a memory leak per NanoGPT
            # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
            return np.memmap(self.data_path, dtype=np.uint16, mode="r")

    def __getitem__(self, idx):
        # Return the underlying datapoint, no random sampling, no worker sharding
        assert 0 <= idx < len(self)
        data = self._get_data()
        x = torch.from_numpy(data[idx : idx + self.sequence_length].astype(np.int64))
        y = torch.from_numpy(
            data[idx + 1 : idx + self.sequence_length + 1].astype(torch.int64)
        )
        return x, y

    def set_step(self, step):
        self.step = step

    def sample_batch(self):
        data = self._get_data()

        if self.with_replacement:
            idxs = self._sample_with_replacement(self.step)
        else:
            idxs = self._sample_without_replacement(self.step)
        self.step += 1

        xy = np.stack([data[i : i + self.sequence_length + 1] for i in idxs]).astype(
            np.int64
        )
        x = torch.from_numpy(xy[:, :-1]).contiguous()
        y = torch.from_numpy(xy[:, 1:]).contiguous()
        return x, y

    def _sample_with_replacement(self, idx):
        # Return an array of token indices of length self.batch_size
        # Sampled with replacement, can get repeats at any time
        seed = self.seed + idx * self.world_size + self.rank
        rng = np.random.default_rng(seed)
        return rng.integers(len(self), self.batch_size)

    def _shuffle_epoch(self, epoch):
        seed = self.seed + epoch
        rng = np.random.default_rng(seed)
        # Drop one sequence to allow different offsets per epoch:
        self.order = rng.permutation((len(self)) // self.sequence_length - 1)
        # Shift all sequences in this epoch by this amount:
        self.epoch_offset = rng.integers(self.sequence_length)
        self.last_epoch = epoch
        self.num_batches_of_seqlen = (
            len(self.order) // self.batch_size
        )  # Drops remainder batch

    def _sample_without_replacement(self, step):
        # Return an array of token indices of length self.batch_size
        # Sampled without replacement, cycle all sequences before potential repeats
        # Sequences are randomly offset in every epoch as well
        batch_idx = self.world_size * step + self.rank
        epoch_length = self.num_batches_of_seqlen

        epoch = batch_idx // epoch_length
        if epoch != self.last_epoch:
            self._shuffle_epoch(epoch)
        epoch_idx = batch_idx % epoch_length

        start = epoch_idx * self.batch_size
        end = start + self.batch_size
        return self.order[start:end] * self.sequence_length + self.epoch_offset

    def num_batches(self):
        if self.with_replacement:
            return self.num_tokens // self.batch_size
        return self.num_batches_of_seqlen

class BenchmarkReader:
    """
    Reader for benchmark validation sets.

    This reader is aligned with the preprocessing in src/data/benchmarks.py:
    - each example is already tokenized with tokenize_with_pad(...)
    - each example length is stored in *.len
    - padding token is GPT2 EOT
    - we never read across example boundaries
    """

    def __init__(
        self,
        data_src,
        lengths,
        batch_size,
        sequence_length,
        keep_in_ram=False,
        pad_token_id=None,
    ):
        if isinstance(data_src, (str, Path)):
            self.data_path = Path(data_src)
            self.keep_in_ram = keep_in_ram
            if keep_in_ram:
                self.data = np.array(
                    np.memmap(self.data_path, dtype=np.uint16, mode="r")
                )
            else:
                self.data = None
        elif isinstance(data_src, (np.ndarray, np.memmap)):
            self.data_path = None
            self.data = data_src
            self.keep_in_ram = True
        else:
            raise TypeError(f"Unsupported data_src type: {type(data_src)}")

        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.step = 0
        self.pad_token_id = int(tknzr.eot_token if pad_token_id is None else pad_token_id)

        # each entry is (chunk_start, example_end)
        self.chunks = self._build_chunks()
        self.num_tokens = len(self._get_data())

    def _get_data(self):
        if self.data is not None:
            return self.data
        return np.memmap(self.data_path, dtype=np.uint16, mode="r")

    def _build_chunks(self):
        """
        Build chunk starts inside each example only.

        benchmark.py already pads each example to a multiple of pad_to_multiple,
        so here we simply split each example into sequence_length-sized chunks
        without ever crossing the example boundary.
        """
        chunks = []
        offsets = np.cumsum(
            np.concatenate(([0], self.lengths[:-1]))
        ).astype(np.int64)

        for off, L in zip(offsets, self.lengths):
            ex_start = int(off)
            ex_end = int(off + L)

            s = ex_start
            while s < ex_end:
                chunks.append((s, ex_end))
                s += self.sequence_length

        return chunks

    def _pad_to_seq_len(self, arr):
        arr = np.asarray(arr, dtype=np.int64)
        if len(arr) == self.sequence_length:
            return arr
        out = np.full(self.sequence_length, self.pad_token_id, dtype=np.int64)
        out[: len(arr)] = arr
        return out

    def _make_xy(self, data, start, ex_end):
        """
        Build x and y strictly within the current example.

        x  = tokens[start : start + T]                  padded with EOT if needed
        y  = tokens[start + 1 : start + T + 1]         padded with EOT if needed

        If the shifted target would cross the example boundary, we pad with EOT
        instead of reading the next example's first token.
        """
        x_end = min(start + self.sequence_length, ex_end)
        y_end = min(start + self.sequence_length + 1, ex_end)

        x = self._pad_to_seq_len(data[start:x_end])
        y = self._pad_to_seq_len(data[start + 1:y_end])

        return x, y

    def set_step(self, step):
        self.step = step

    def num_batches(self):
        return math.ceil(len(self.chunks) / self.batch_size)

    def sample_batch(self):
        start_idx = self.step * self.batch_size
        end_idx = min((self.step + 1) * self.batch_size, len(self.chunks))
        batch_chunks = self.chunks[start_idx:end_idx]
        self.step += 1

        data = self._get_data()

        xs = []
        ys = []
        for start, ex_end in batch_chunks:
            x, y = self._make_xy(data, start, ex_end)
            xs.append(x)
            ys.append(y)

        x = torch.from_numpy(np.stack(xs)).contiguous()
        y = torch.from_numpy(np.stack(ys)).contiguous()
        return x, y

def get_benchmark_readers(args):
    readers = {}

    benchmark_kwargs = {
        "num_proc": getattr(args, "benchmark_num_proc", 10),
        "return_torch": False,
        "pad_to_multiple": getattr(args, "benchmark_pad_to_multiple", 1024),
    }

    for name in args.benchmark_eval_tasks:
        task_data = get_benchmark_task(name, **benchmark_kwargs)
        readers[name] = BenchmarkReader(
            data_src=task_data["val"],
            lengths=task_data["val_len"],
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            keep_in_ram=args.data_in_ram,
            pad_token_id=tknzr.eot_token,
        )
    return readers