from multiprocessing.shared_memory import SharedMemory
from typing import Tuple

import numpy as np

from mpaio.worker import Worker


class ConcatStrWorker(Worker["str"]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._result = ""

    def consume_callback(self, processed_items: str):
        self._result += processed_items

    @staticmethod
    def process(
        shm_name: str,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        start_idx: int,
        end_idx: int,
    ) -> str:
        shm = SharedMemory(shm_name)
        data: np.ndarray = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
        concatenated_label = ""
        for i in range(start_idx, end_idx):
            label = data[i]
            concatenated_label += label
        return concatenated_label

    @property
    def result(self):
        return self._result
