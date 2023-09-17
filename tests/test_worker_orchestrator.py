import asyncio
import datetime
import math
from collections import defaultdict
from typing import Generic

import anyio
import numpy as np
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
from concurrent.futures import Executor

from mpaio.data_iterator import DataIterator
from mpaio.item_type import ItemT
from mpaio.worker import Worker
from mpaio.worker_orchestrator import WorkerOrchestrator


@pytest.fixture
def mock_send_channel():
    send_channel = AsyncMock()
    send_channel.__aenter__.return_value = send_channel
    send_channel.__aexit__.return_value = None
    send_channel.clone = MagicMock(return_value=send_channel)
    return send_channel


@pytest.fixture
def mock_receive_channel():
    receive_channel = MagicMock()
    return receive_channel


@pytest.fixture
def create_mock_memory_object_stream(mock_send_channel, mock_receive_channel):
    class mock_create_memory_object_stream(Generic[ItemT]):
        def __new__(cls):
            return mock_send_channel, mock_receive_channel

    return mock_create_memory_object_stream


@pytest.fixture
def mock_executor():
    return MagicMock(Executor)


@pytest.fixture
def mock_workers():
    return [Mock(Worker)]


@pytest.fixture
def make_orchestrator(mock_executor, mock_workers):
    def orchestrator(workers=None, monitor=False):
        if workers is None:
            workers = mock_workers
        return WorkerOrchestrator(mock_executor, workers, monitor_cpu_usage=monitor)

    return orchestrator


@pytest.mark.anyio
async def test_wrapper(make_orchestrator):
    orchestrator = make_orchestrator()
    fut = asyncio.Future()
    fut.set_result("foo")
    send_channel = AsyncMock()
    send_channel.__aenter__.return_value = send_channel
    send_channel.__aexit__.return_value = None

    await orchestrator.wrapper(fut, send_channel)

    send_channel.__aenter__.assert_called_once()
    send_channel.send.assert_called_once_with("foo")


@pytest.mark.anyio
async def test_cpu_percent(make_orchestrator):
    orchestrator = make_orchestrator()
    df_data = defaultdict(list)

    # Patch the datetime module to return a deterministic value
    fixed_utcnow = datetime.datetime(2019, 9, 13, 12, 0, 0)
    mock_datetime = MagicMock()
    mock_datetime.utcnow.return_value = fixed_utcnow

    # Mock the psutil.cpu_percent call to return a specific value
    mock_cpu_percent = Mock(return_value=[10.0, 20.0, 30.0])
    with patch("psutil.cpu_percent", mock_cpu_percent), patch(
        "datetime.datetime", mock_datetime
    ):
        async with anyio.create_task_group() as tg:
            tg.start_soon(orchestrator.cpu_percent, df_data)
            while len(df_data) == 0:
                await asyncio.sleep(0)
            tg.cancel_scope.cancel()

    # Check that df_data contains the expected CPU percentages and fixed_utcnow
    assert df_data == {
        "core_0": [10.0],
        "core_1": [20.0],
        "core_2": [30.0],
        "time": [fixed_utcnow],
    }


@pytest.mark.anyio
async def test_run_no_monitoring(
    make_orchestrator, create_mock_memory_object_stream, mock_send_channel
):
    worker_1 = MagicMock()
    worker_2 = MagicMock()

    chunk_size_1 = 2
    size_1 = 10
    iterator1 = DataIterator(
        shm_name="test_shm",
        chunk_size=chunk_size_1,
        size_of_data=size_1,
        shm_shape=(10,),
        data_type=np.int32,
    )
    chunk_size_2 = 4
    size_2 = 10
    iterator2 = DataIterator(
        shm_name="test_shm2",
        chunk_size=4,
        size_of_data=10,
        shm_shape=(10,),
        data_type=np.int32,
    )

    total_number_of_iterations = math.ceil(size_1 / chunk_size_1) + math.ceil(
        size_2 / chunk_size_2
    )

    worker_1.data_iterator = iterator1
    worker_1.consumer = AsyncMock()

    worker_2.data_iterator = iterator2
    worker_2.consumer = AsyncMock()

    workers = [worker_1, worker_2]
    orchestrator = make_orchestrator(workers=workers)
    mock_executor = orchestrator._executor

    mock_executor.__enter__.return_value = mock_executor
    mock_executor.__exit__.return_value = None
    future = asyncio.Future()
    mock_executor.submit = MagicMock(return_value=future)
    future.set_result("foo")

    with patch("anyio.create_memory_object_stream", create_mock_memory_object_stream):
        result = await orchestrator.run()

    assert result == {}  # Assuming no CPU monitoring

    mock_executor.__enter__.assert_called()
    mock_executor.__exit__.assert_called()
    mock_executor.submit.assert_called()
    mock_send_channel.send.assert_has_awaits([call("foo")] * total_number_of_iterations)
    worker_1.consumer.assert_awaited()
    worker_2.consumer.assert_awaited()


@pytest.mark.anyio
async def test_run_monitoring(
    monkeypatch, make_orchestrator, mock_send_channel, create_mock_memory_object_stream
):
    worker_1 = MagicMock()
    worker_2 = MagicMock()

    chunk_size_1 = 2
    size_1 = 10
    iterator1 = DataIterator(
        shm_name="test_shm",
        chunk_size=chunk_size_1,
        size_of_data=size_1,
        shm_shape=(10,),
        data_type=np.int32,
    )
    chunk_size_2 = 4
    size_2 = 10
    iterator2 = DataIterator(
        shm_name="test_shm2",
        chunk_size=4,
        size_of_data=10,
        shm_shape=(10,),
        data_type=np.int32,
    )

    total_number_of_iterations = math.ceil(size_1 / chunk_size_1) + math.ceil(
        size_2 / chunk_size_2
    )

    worker_1.data_iterator = iterator1
    worker_1.consumer = AsyncMock()

    worker_2.data_iterator = iterator2
    worker_2.consumer = AsyncMock()

    workers = [worker_1, worker_2]
    orchestrator = make_orchestrator(workers=workers, monitor=True)
    mock_executor = orchestrator._executor

    mock_executor.__enter__.return_value = mock_executor
    mock_executor.__exit__.return_value = None
    future = asyncio.Future()
    mock_executor.submit = MagicMock(return_value=future)
    future.set_result("foo")

    # Patch the datetime module to return a deterministic value
    fixed_utcnow = datetime.datetime(2019, 9, 13, 12, 0, 0)
    mock_datetime = MagicMock()
    mock_datetime.utcnow.return_value = fixed_utcnow

    mock_cpu_util = [10.0, 20.0, 30.0]
    mock_cpu_percent = Mock(return_value=mock_cpu_util)

    mock_memory_object_stream = create_mock_memory_object_stream

    with patch("anyio.create_memory_object_stream", mock_memory_object_stream), patch(
        "psutil.cpu_percent", mock_cpu_percent
    ), patch("datetime.datetime", mock_datetime):
        result = await orchestrator.run()

    expected_result = defaultdict(list)
    for i, cpu_util in enumerate(mock_cpu_util):
        expected_result[f"core_{i}"].append(cpu_util)
    expected_result["time"] = [fixed_utcnow]

    assert result == expected_result

    mock_executor.__enter__.assert_called()
    mock_executor.__exit__.assert_called()
    mock_executor.submit.assert_called()
    mock_send_channel.send.assert_has_awaits([call("foo")] * total_number_of_iterations)
    worker_1.consumer.assert_awaited()
    worker_2.consumer.assert_awaited()
