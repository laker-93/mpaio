from multiprocessing.shared_memory import SharedMemory

import numpy as np

from mpaio.core.worker_orchestrator import Worker


class ConcatStrWorker(Worker['str']):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._result = ''

    def consume_callback(self, processed_items: str):
        print(f'worker got result {processed_items}')
        self._result += processed_items

    @staticmethod
    def process(shm_name: str, shape, dtype, start_idx, end_idx) -> str:
        shm = SharedMemory(shm_name)
        data = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
        concatenated_label = ''
        for i in range(start_idx, end_idx):
            label = data[i]
            concatenated_label += label
        return concatenated_label
