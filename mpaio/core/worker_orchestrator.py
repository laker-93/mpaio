import asyncio
import datetime

import matplotlib
import matplotlib.pyplot as plt
import time
from collections import defaultdict

import pandas as pd
import psutil
import functools
import logging
from typing import List, Dict

import anyio
from anyio.streams.memory import MemoryObjectSendStream
from anyio import to_process

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

    @staticmethod
    async def cpu_percent(df_data: Dict):
        while True:
            res = psutil.cpu_percent(None, percpu=True)
            for i, core_utilisation in enumerate(res):
                df_data[f'core_{i}'].append(core_utilisation)
            df_data['time'].append(datetime.datetime.utcnow())
            await asyncio.sleep(0.2)

    async def run(self):
        loop = asyncio.get_running_loop()
        df_data = defaultdict(list)
        with self._executor as executor:
            async with anyio.create_task_group() as monitoring_tg:
                monitoring_tg.start_soon(self.cpu_percent, df_data)
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
                    print('cancelling main')
                print('cancelling monitoring')
                monitoring_tg.cancel_scope.cancel()
        #plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M:%S.%f"))
        df = pd.DataFrame(df_data)
        df = df.set_index('time')
        df.index = pd.to_timedelta(df.index - df.index[0], unit='milliseconds')
        #df.index = pd.to_timedelta(df.index, unit='seconds')
        plot = df.plot(y=df.columns, title='cpu usage', xlabel='time')
        plt.show()
