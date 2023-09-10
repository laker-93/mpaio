from multiprocessing.shared_memory import SharedMemory

import numpy as np

from mpaio.core.worker_orchestrator import Worker


class AddIntWorker(Worker['int']):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._result = 0

    def consume_callback(self, processed_items: int):
        print(f'worker got result {processed_items}')
        self._result += processed_items

    @staticmethod
    def process(shm_name: str, shape, dtype, start_idx, end_idx) -> int:
        shm = SharedMemory(shm_name)
        data = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
        print(
            f'worker running from start_idx: value ({start_idx}: {data[start_idx]}) to end_idx: value ({end_idx - 1}: {data[end_idx - 1]})')
        result = 0
        for i in range(start_idx, end_idx):
            result += data[i]
        print(f'worker got result {result}')
        return result
