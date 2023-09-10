import asyncio
import functools
import logging
import os
from abc import ABC, abstractmethod
from typing import Generic, List

import anyio
import psutil

from mpaio.core.generic_types import ItemT

logger = logging.getLogger(__name__)


class Worker(Generic[ItemT], ABC):

    def __init__(self, data_iterator, send_channel, receive_channel):
        self._data_iterator = data_iterator
        self._send_channel = send_channel
        self._receive_channel = receive_channel

    # has to be static since this will be run in a separate process and therefore won't have access to the class instance
    @staticmethod
    @abstractmethod
    def process(shm_name: str, shape, dtype, start_idx, end_idx) -> ItemT:
        pass

    @staticmethod
    def _process(method_to_run, shm_name: str, shape, dtype, start_idx, end_idx) -> ItemT:
        process = psutil.Process()
        print(process.memory_info().rss)
        print(f'pid {os.getpid()}')
        res = method_to_run(shm_name, shape, dtype, start_idx, end_idx)
        return res

    @abstractmethod
    def consume_callback(self, processed_items: ItemT):
        pass

    async def consumer(self, receive_channel):
        async with receive_channel:
            async for res in receive_channel:
                self.consume_callback(res)


class WorkerOrchestrator:
    def __init__(self, executor, workers: List[Worker]):
        self._executor = executor
        self._workers = workers

    @staticmethod
    async def wrapper(fut, send_channel):
        async with send_channel:
            res = await fut
            await send_channel.send(res)

    async def run(self):
        loop = asyncio.get_running_loop()
        with self._executor as executor:
            async with anyio.create_task_group() as tg:
                for worker in self._workers:
                    data = worker._data_iterator
                    send_channel = worker._send_channel
                    receive_channel = worker._receive_channel
                    async with send_channel, receive_channel:
                        tg.start_soon(worker.consumer, receive_channel.clone())
                        for start_idx, end_idx in data:
                            fut = loop.run_in_executor(
                                executor,
                                functools.partial(worker._process,
                                                  worker.process,
                                                  data.shm_name,
                                                  data.shm_shape,
                                                  data.dtype,
                                                  start_idx,
                                                  end_idx)
                            )
                            tg.start_soon(self.wrapper, fut, send_channel.clone())
