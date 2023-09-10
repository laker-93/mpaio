import asyncio
import functools
import logging
from typing import List

import anyio
from anyio.streams.memory import MemoryObjectSendStream

from mpaio.core.item_type import ItemT
from mpaio.core.worker import Worker

logger = logging.getLogger(__name__)


class WorkerOrchestrator:
    def __init__(self, executor, workers: List[Worker]):
        self._executor = executor
        self._workers = workers

    @staticmethod
    async def wrapper(fut: asyncio.Future, send_channel: MemoryObjectSendStream[ItemT]):
        async with send_channel:
            res = await fut
            await send_channel.send(res)

    async def run(self):
        loop = asyncio.get_running_loop()
        with self._executor as executor:
            async with anyio.create_task_group() as tg:
                for worker in self._workers:
                    data = worker.data_iterator
                    send_channel = worker.send_channel
                    receive_channel = worker.receive_channel
                    async with send_channel, receive_channel:
                        tg.start_soon(worker.consumer, receive_channel.clone())
                        for start_idx, end_idx in data:
                            fut = loop.run_in_executor(
                                executor,
                                functools.partial(worker.process,
                                                  data.shm_name,
                                                  data.shm_shape,
                                                  data.dtype,
                                                  start_idx,
                                                  end_idx)
                            )
                            tg.start_soon(self.wrapper, fut, send_channel.clone())
