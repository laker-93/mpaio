from multiprocessing.shared_memory import SharedMemory
from typing import Tuple

import numpy as np

from mpaio.worker import Worker


class AddIntWorker(Worker["int"]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._result = 0

    def consume_callback(self, processed_items: int):
        self._result += processed_items

    @staticmethod
    def process(
        shm_name: str,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        start_idx: int,
        end_idx: int,
    ) -> int:
        shm = SharedMemory(shm_name)
        data: np.ndarray = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
        result = 0
        for i in range(start_idx, end_idx):
            result += data[i]
        return result

    @property
    def result(self):
        return self._result
