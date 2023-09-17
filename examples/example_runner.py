import datetime
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.managers import SharedMemoryManager

import anyio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from workers.add_int_worker import AddIntWorker
from workers.concat_str_worker import ConcatStrWorker
from mpaio.data_iterator import DataIterator
from mpaio.worker_orchestrator import WorkerOrchestrator


def setup_concat_str_worker(
    manager: SharedMemoryManager, n_workers: int
) -> ConcatStrWorker:
    data_string = np.array([f"foo_{i}" for i in range(100)])
    shm_strings = manager.SharedMemory(data_string.nbytes)
    shm_strings_data: np.ndarray = np.ndarray(
        shape=data_string.shape, dtype=data_string.dtype, buffer=shm_strings.buf
    )
    # data copied to the shared memory is done so by python internals using pickle. Writing directly to the shared
    # memory is not supported so have to copy it here.
    np.copyto(shm_strings_data, data_string)

    str_data_iterator = DataIterator(
        shm_name=shm_strings.name,
        chunk_size=20,
        data_type=shm_strings_data.dtype,
        size_of_data=shm_strings_data.size,
        shm_shape=shm_strings_data.shape,
    )

    return ConcatStrWorker(data_iterator=str_data_iterator)


def setup_add_int_worker(manager: SharedMemoryManager, n_workers: int) -> AddIntWorker:
    start = 1
    end = 100_000_001
    chunk_size = (end - start) // n_workers
    data_nums = np.array(list(range(start, end)))
    shm_nums = manager.SharedMemory(data_nums.nbytes)
    shm_nums_data: np.ndarray = np.ndarray(
        shape=data_nums.shape, dtype=data_nums.dtype, buffer=shm_nums.buf
    )
    # data copied to the shared memory is done so by python internals using pickle. Writing directly to the shared
    # memory is not supported so have to copy it here.
    np.copyto(shm_nums_data, data_nums)

    num_data_iterator = DataIterator(
        shm_name=shm_nums.name,
        chunk_size=chunk_size,
        data_type=shm_nums_data.dtype,
        size_of_data=shm_nums_data.size,
        shm_shape=shm_nums_data.shape,
    )

    return AddIntWorker(data_iterator=num_data_iterator)


async def runner():
    n_workers = 6
    executor = ProcessPoolExecutor(
        mp_context=multiprocessing.get_context("spawn"), max_workers=n_workers
    )

    with SharedMemoryManager() as manager:
        worker_add = setup_add_int_worker(manager, n_workers)
        worker_str = setup_concat_str_worker(manager, n_workers)
        mp_manager = WorkerOrchestrator(
            executor, workers=[worker_add, worker_str], monitor_cpu_usage=True
        )
        start_time = datetime.datetime.utcnow()
        df_data = await mp_manager.run()
        end_time = datetime.datetime.utcnow()
        total_time = (end_time - start_time).total_seconds()
        print(f"got result {worker_add.result} in {total_time}")
        print(f"got result {worker_str.result} in {total_time}")
        if df_data:
            _plot_df(df_data)
        # print(f'got result {worker_str._result} in {total_time}')


def _plot_df(df_data):
    df = pd.DataFrame(df_data)
    df = df.set_index("time")
    df.index = pd.to_timedelta(df.index - df.index[0], unit="milliseconds")
    df.plot(y=df.columns, title="cpu usage per cpu", xlabel="time", ylabel="cpu usage")
    plt.show()


def main():
    anyio.run(runner)


if __name__ == "__main__":
    main()
