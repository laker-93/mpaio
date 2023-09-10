import datetime
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.managers import SharedMemoryManager

import anyio
import numpy as np

from examples.workers.add_int_worker import AddIntWorker
from examples.workers.concat_str_worker import ConcatStrWorker
from mpaio.core.data_iterator import DataIterator
from mpaio.core.worker_orchestrator import WorkerOrchestrator

n_workers = 8


async def runner():
    executor = ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn"))

    start = 1
    end = 100_000_001
    chunk_size = (end - start) // n_workers
    data_nums = np.array(list(range(start, end)))
    data_string = np.array([f'foo_{i}' for i in range(100)])
    with SharedMemoryManager() as manager:
        shm_nums = manager.SharedMemory(data_nums.nbytes)
        shm_nums_data = np.ndarray(shape=data_nums.shape, dtype=data_nums.dtype, buffer=shm_nums.buf)
        shm_strings = manager.SharedMemory(data_string.nbytes)
        shm_strings_data = np.ndarray(shape=data_string.shape, dtype=data_string.dtype, buffer=shm_strings.buf)
        # data copied to the shared memory is done so by python internals using pickle. Writing directly to the shared
        # memory is not supported so have to copy it here.
        np.copyto(shm_nums_data, data_nums)
        np.copyto(shm_strings_data, data_string)
        del data_nums
        del data_string
        start_time = datetime.datetime.utcnow()

        num_data_iterator = DataIterator(
            shm_name=shm_nums.name,
            chunk_size=chunk_size,
            dtype=shm_nums_data.dtype,
            size_of_data=shm_nums_data.size,
            shm_shape=shm_nums_data.shape
        )

        send_channel, receive_channel = anyio.create_memory_object_stream[int]()
        worker_add = AddIntWorker(
            data_iterator=num_data_iterator,
            send_channel=send_channel,
            receive_channel=receive_channel
        )

        str_data_iterator = DataIterator(
            shm_name=shm_strings.name,
            chunk_size=20,
            dtype=shm_strings_data.dtype,
            size_of_data=shm_strings_data.size,
            shm_shape=shm_strings_data.shape
        )

        send_channel, receive_channel = anyio.create_memory_object_stream[str]()
        worker_str = ConcatStrWorker(
            data_iterator=str_data_iterator,
            send_channel=send_channel,
            receive_channel=receive_channel
        )
        mp_manager = WorkerOrchestrator(executor, workers=[worker_add, worker_str])

        res = await mp_manager.run()
        end_time = datetime.datetime.utcnow()
        total_time = (end_time - start_time).total_seconds()
        print(f'got result {worker_add._result} in {total_time}')
        print(f'got result {worker_str._result} in {total_time}')


def main():
    anyio.run(runner)


if __name__ == '__main__':
    main()
