from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator

from numpy import dtype


@dataclass(frozen=True)
class DataIterator:
    shm_name: str
    chunk_size: int
    size_of_data: int
    shm_shape: tuple[int, ...]
    data_type: dtype
    start_idx: int = 0

    def __iter__(self) -> Iterator[tuple[int, int]]:
        start = self.start_idx
        while start < self.size_of_data:
            end = min(start + self.chunk_size, self.size_of_data)
            yield start, end
            start += self.chunk_size
