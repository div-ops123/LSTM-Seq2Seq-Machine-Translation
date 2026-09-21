"""Builds padded ID tensors and DataLoaders from the parallel corpus files.

parameterized by whatever vocabs and MAX_LENGTH the EDA decides on, rather
than assuming a fixed short-sentence filter (this project's Milestone 5
needs long/very-long source sentences to remain in the data).
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from src.vocab import Vocab


def load_pairs(src_path: str, tgt_path: str, normalize_src, normalize_tgt) -> list[tuple[str, str]]:
    with open(src_path, encoding="utf-8") as f_src, open(tgt_path, encoding="utf-8") as f_tgt:
        src_lines = f_src.readlines()
        tgt_lines = f_tgt.readlines()
    if len(src_lines) != len(tgt_lines):
        raise ValueError(
            f"{src_path} has {len(src_lines)} lines but {tgt_path} has {len(tgt_lines)}"
        )
    return [
        (normalize_src(s), normalize_tgt(t))
        for s, t in zip(src_lines, tgt_lines)
    ]


def encode_pairs(
    pairs: list[tuple[str, str]],
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(pairs)
    input_ids = np.zeros((n, max_length), dtype=np.int64)
    target_ids = np.zeros((n, max_length), dtype=np.int64)
    for i, (src, tgt) in enumerate(pairs):
        input_ids[i] = src_vocab.sentence_to_ids(src, max_length)
        target_ids[i] = tgt_vocab.sentence_to_ids(tgt, max_length)
    return input_ids, target_ids


def make_dataloader(
    input_ids: np.ndarray,
    target_ids: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    device: str = "cpu",
) -> DataLoader:
    data = TensorDataset(
        torch.as_tensor(input_ids, dtype=torch.long, device=device),
        torch.as_tensor(target_ids, dtype=torch.long, device=device),
    )
    if shuffle:
        sampler = RandomSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=batch_size)
    return DataLoader(data, batch_size=batch_size, shuffle=False)
