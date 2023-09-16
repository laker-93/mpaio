import pytest
from unittest.mock import Mock
from anyio.streams.memory import MemoryObjectSendStream, MemoryObjectReceiveStream
from numpy.core.multiarray import dtype
from mpaio.data_iterator import DataIterator
from mpaio.worker import Worker
import numpy as np


class AsyncContextManagerMock(Mock):
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        pass


class TestWorker(Worker['str']):
    @staticmethod
    def process(
        shm_name: str,
        shape: tuple[int, ...],
        dtype: dtype,
        start_idx: int,
        end_idx: int,
    ) -> str:
        return 'foo'

    def consume_callback(self, processed_items: str):
        pass


@pytest.fixture
def data_iterator():
    return DataIterator(
        shm_name="test_shm",
        chunk_size=3,
        size_of_data=10,
        shm_shape=(10,),
        dtype=dtype(np.int32),
    )


@pytest.fixture
def send_channel():
    return Mock(spec=MemoryObjectSendStream)


@pytest.fixture
def receive_channel():
    # Create a mock with added functionality for adding items and supporting async iteration
    mock_receive_channel = AsyncContextManagerMock(spec=MemoryObjectReceiveStream)
    mock_receive_channel._items = []  # Initialize a list to store items

    # Add a method to add items to the channel
    def add_item(item):
        mock_receive_channel._items.append(item)

    mock_receive_channel.add_item = add_item

    async def _aiter(self):
        for item in mock_receive_channel._items:
            yield item

    mock_receive_channel.__aiter__ = _aiter

    return mock_receive_channel


def test_process_mocked(data_iterator, send_channel, receive_channel):
    worker = TestWorker(data_iterator, send_channel, receive_channel)
    worker.process = Mock(return_value="processed_item")

    result = worker.process("shm_name", (1, 2), dtype(np.int32), 0, 3)

    assert result == "processed_item"


@pytest.mark.anyio
async def test_consume_callback_mocked(data_iterator, send_channel, receive_channel):
    receive_channel.add_item("item1")
    worker = TestWorker(data_iterator, send_channel, receive_channel)
    worker.consume_callback = Mock()
    await worker.consumer(receive_channel)
    worker.consume_callback.assert_called_once_with("item1")
