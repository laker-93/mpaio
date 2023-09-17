from abc import ABC, abstractmethod
from typing import Generic, Tuple

from anyio.streams.memory import MemoryObjectReceiveStream
from numpy.core.multiarray import dtype

from mpaio.data_iterator import DataIterator
from mpaio.item_type import ItemT


class Worker(Generic[ItemT], ABC):
    def __init__(self, data_iterator: DataIterator):
        self._data_iterator = data_iterator

    @property
    def data_iterator(self):
        return self._data_iterator

    # has to be static since this will be run in a separate process and therefore won't have access to the class
    # instance
    @staticmethod
    @abstractmethod
    def process(
        shm_name: str,
        shape: Tuple[int, ...],
        dtype: dtype,
        start_idx: int,
        end_idx: int,
    ) -> ItemT:
        pass

    @abstractmethod
    def consume_callback(self, processed_items: ItemT):
        pass

    async def consumer(self, receive_channel: MemoryObjectReceiveStream[ItemT]):
        async with receive_channel:
            async for res in receive_channel:
                self.consume_callback(res)
