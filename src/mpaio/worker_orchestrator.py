import asyncio
import datetime
import functools
import logging
from collections import defaultdict
from concurrent.futures import Executor
from typing import List, Dict, Any

import anyio
import psutil
from anyio.streams.memory import MemoryObjectSendStream

from mpaio.item_type import ItemT
from mpaio.worker import Worker

logger = logging.getLogger(__name__)


class WorkerOrchestrator:
    def __init__(
        self, executor: Executor, workers: List[Worker], monitor_cpu_usage: bool
    ):
        self._executor = executor
        self._workers = workers
        self._monitor_cpu_usage = monitor_cpu_usage

    @staticmethod
    async def wrapper(fut: asyncio.Future, send_channel: MemoryObjectSendStream[ItemT]):
        """
        Wrapper that can ber used to await the future as a task and send the result of the completed future over the
        send channel.
        :param fut:
        :param send_channel:
        :return:
        """
        async with send_channel:
            res = await fut
            await send_channel.send(res)

    @staticmethod
    async def cpu_percent(df_data: Dict):
        """
        Task to monitor cpu percentage per src and construct a dictionary of results.
        :param df_data:
        :return:
        """
        while True:
            res = psutil.cpu_percent(None, percpu=True)
            for i, core_utilisation in enumerate(res):
                df_data[f"core_{i}"].append(core_utilisation)
            df_data["time"].append(datetime.datetime.utcnow())
            await asyncio.sleep(0.2)

    async def run(self) -> Dict:
        loop = asyncio.get_running_loop()
        df_data: Dict = defaultdict(list)

        with self._executor as executor:
            async with anyio.create_task_group() as monitoring_tg:
                if self._monitor_cpu_usage:
                    monitoring_tg.start_soon(self.cpu_percent, df_data)
                async with anyio.create_task_group() as tg:
                    for worker in self._workers:
                        data = worker.data_iterator
                        # The type sent over these memory channels is the type returned by the worker's process method
                        # using introspection to get this type dynamically is not well supported yet by Python so for
                        # now just use 'Any' as the type.
                        (
                            send_channel,
                            receive_channel,
                        ) = anyio.create_memory_object_stream[Any]()
                        async with send_channel, receive_channel:
                            tg.start_soon(worker.consumer, receive_channel.clone())
                            for start_idx, end_idx in data:
                                fut = loop.run_in_executor(
                                    executor,
                                    functools.partial(
                                        worker.process,
                                        data.shm_name,
                                        data.shm_shape,
                                        data.data_type,
                                        start_idx,
                                        end_idx,
                                    ),
                                )
                                tg.start_soon(self.wrapper, fut, send_channel.clone())
                monitoring_tg.cancel_scope.cancel()
        return df_data
