from abc import ABC, abstractmethod
from typing import Generic

from anyio.streams.memory import MemoryObjectSendStream, MemoryObjectReceiveStream
from numpy.core.multiarray import dtype

from src.data_iterator import DataIterator
from src.item_type import ItemT


class Worker(Generic[ItemT], ABC):

    def __init__(
            self,
            data_iterator: DataIterator,
            send_channel: MemoryObjectSendStream[ItemT],
            receive_channel: MemoryObjectReceiveStream[ItemT]
    ):
        self._data_iterator = data_iterator
        self._send_channel = send_channel
        self._receive_channel = receive_channel

    @property
    def data_iterator(self):
        return self._data_iterator

    @property
    def send_channel(self):
        return self._send_channel

    @property
    def receive_channel(self):
        return self._receive_channel

    # has to be static since this will be run in a separate process and therefore won't have access to the class instance
    @staticmethod
    @abstractmethod
    def process(shm_name: str, shape: tuple[int, ...], dtype: dtype, start_idx: int, end_idx: int) -> ItemT:
        pass

    @abstractmethod
    def consume_callback(self, processed_items: ItemT):
        pass

    async def consumer(self, receive_channel: MemoryObjectReceiveStream[ItemT]):
        async with receive_channel:
            async for res in receive_channel:
                self.consume_callback(res)
