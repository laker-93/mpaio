import pytest
import numpy as np

from mpaio.data_iterator import DataIterator


# Define test cases
@pytest.mark.parametrize(
    "chunk_size, size_of_data, expected_chunks",
    [
        (
            3,
            10,
            [(0, 3), (3, 6), (6, 9), (9, 10)],
        ),  # Test with chunk size smaller than data
        (5, 10, [(0, 5), (5, 10)]),  # Test with chunk size equal to data
        (7, 10, [(0, 7), (7, 10)]),  # Test with chunk size larger than data
    ],
)
def test_data_iterator_iteration(chunk_size, size_of_data, expected_chunks):
    iterator = DataIterator(
        shm_name="test_shm",
        chunk_size=chunk_size,
        size_of_data=size_of_data,
        shm_shape=(10,),
        data_type=np.int32,
    )

    chunks = list(iterator)

    assert chunks == expected_chunks


def test_data_iterator_start_idx():
    """
    Additional test to ensure start_idx is considered
    :return:
    """
    iterator = DataIterator(
        shm_name="test_shm",
        chunk_size=4,
        size_of_data=12,
        shm_shape=(12,),
        data_type=np.int32,
        start_idx=2,
    )

    chunks = list(iterator)

    assert chunks == [(2, 6), (6, 10), (10, 12)]
