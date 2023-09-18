# MPAIO

## Overview

MPAIO is a library for parallel processing a numpy array using a pool of workers, each running on a separate process. It
performs the processing asynchronously so none of the work in starting the workers, or collecting their results when
finished, blocks. It is a generalised library inspired by Lukasz Langa's PyCon 2023 talk: 'Working around the GIL with
asyncio'.

Each worker handles processing a chunk of the array and MPAIO coordinates giving the results back to the user.
MPAIO expects the array to be processed to be available in shared memory and to remain constant.
MPAIO internally uses Python std library `ProcessPoolExecutor` to run the workers:
https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor-example
MPAIO uses anyio to do the asynchronous scheduling.

MPAIO is composed of:
    - a `DataIterator` class
        - encapsulates meta-data of the shared memory buffer and logic of how to partition the array amongst the workers.
    - an abstract `Worker` class
        - defines a template for a worker and the logic to process a chunk of the data.
        - a `Worker` is constructed with the `DataIterator` object that it will be processing.
    - a `WorkerOrchestrator` class
        - runs the workers in the executor and makes the results from the sub processes available in the main process.

## How to use

1. Create a numpy array (or multiple arrays) containing the data that needs to be processed and copy the array(s) in to
a shared memory block(s).

2. Create a `ProcessPoolExecutor` that will be used to create sub processes to run the `Worker`s on. Note that any
executor can be used here that implements the concurrent future `Executor` base class. However, due to the GIL, only
use of a `ProcessPoolExecutor` will result in the work being executed in parallel.

3. Define a `DataIterator`, containing the meta-data for the shared memory block and the logic for how the shared memory
block should be partitioned amongst the workers. One for each shared memory block.

4. Define a worker that implements the abstract MPAIO `Worker` base class. Pass the `DataIterator` in to the constructor
of the `Worker`. One for each shared memory block.
    - The `process` method will be run in a separate process. It is passed the meta data of the shared memory array as
   well as the start and end index that it will be responsible for processing. It must recreate the numpy array from the
   meta data, process the slice of the array and return the result. The returned result must be _pickleable_.
    - The `process_callback` method will be run in the main process. It will receive the processed data that the
   `process` method returned.

5. Construct a `WorkerOrchestrator` and pass in the executor and the list of `Worker`s.
    - The `WorkerOrchestrator` will allocate each partition of data to a separate invocation of the `process` method of
   the `Worker`. This will be scheduled to run on a free worker from the executor pool.
    - When a worker has finished, the `process_callback` method of the `Worker` is called with the processed chunk of
   data.
    - Optionally monitor the CPU usage when running the orchestrator by setting `monitor_cpu_usage` to True. When set,
   this will return a dictionary with the CPU utilisation for each core suitable for creating a time series Panda's
   DataFrame for (see `examples/example_runner.py`).

6. run the `run` coroutine of the `WorkerOrchestrator` asynchronously.

For optimum performance, all cores of the system should be utilised with the total number of partitions of data matching
the number of available cores. If the paritions are too large, then there will be idle cores. Conversely, if the
partitions are too small then there will be unnecessary overheads from workers starting and stopping multiple times.

MPAIO is designed using dependency injection, so the executor and shared memory must be created in the user code and
injected in when constructing the `WorkerOrchestrator`.


## Demo

An example is included in `examples/` that sets up two shared memory arrays, one containing strings, the other
containing integers. For each of these arrays, `DataIterator` is created defining the meta-data for the shared memory
and logic for how the array should be batched. A `AddIntWorker` is defined with the logic how to process a chunk of the
integer arrays. A `ConcatStrWorker` is defined with the logic of how to process a chunk of the string array. The workers
themselves are for demonstrative purposes - they implement some arbitrary CPU intensive operations.

To run the examples:

1. git clone the repo
    a. `git clone https://github.com/laker-93/mpaio.git`
2. create a new venv and activate
    a. `python3 -m venv venv`
    b. `source venv/bin/activate`
3. pip install
    a. `pip install mpaio`
4. pip install the extra dependencies to run the examples
    a. `pip install mpaio[examples]`
    b. `pip install mpaio'[examples]'` (on MacOS)
5. run the example
    a. `python examples/example_runner.py`

This will also produce a plot of the CPU utilisation from running the examples. You can change `n_workers` in 
`example_runner.py` to see the effect os utilising fewer/more cores.

## Implementation Notes
Use data structures created by multiprocess
manager: https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Manager
if needing to coordinate both reading and writing from child to parent processes.
If reads and writes are atomics and do not need to be coordinated, then can simply use a shared memory block which will
provide faster access.

Trick for speed is to only pass small amount of data in to sub processes and recreate
full structures within sub process.
Natural choice when wanting to share say a list is to use the list created by the mp
manager that can be shared between processes. This will be slow but necessary if your
child/parent process is writing to the shared memory dynamically.

Seems tempting to implement as a decorator but this design won't work well when orchestrating multiple workers
with different processing requirements. There's also issues with pickle when attempting to pickle a decorated method.

Since some numpy calls and third party libraries release the GIL under the hood, performance benefits can be seen from
using multithreading.

Option when designing this to register the worker functions using a decorator e.g.

@run_in_subprocess(process_manager)
def worker1(data) -> int:
...

@run_in_subprocess(process_manager)
def worker2(data) -> str:
...

worker1(data) # causes process manager to register the worker - won't run yet
worker2(data) # causes process manager to register the worker - won't run yet
await process_manager.run() # runs all registered workers

however this violates the principle of least surprise. namely it is suprising that calling `worker()` won't run the
worker until the process manager is run.

Uses structured concurrency (anyio) for TaskGroup like task management without having to restrict to Python 3.11.

Use chatgpt for writing unit tests. This is a good litmus test for having small modular classes. ChatGPT generates
excellent tests for small, well designed simple classes whereas it struggles to test complex spaghetti classes.

Use a mix of anyio and asyncio
anyio - excellent library for structured concurrency, gives you task groups without having to be on Python 3.11. It does
not yet have support for synchronisation primitives for multi processing.
asyncio - run concurrent executor within asyncio.