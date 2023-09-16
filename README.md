# MPAIO

MPAIO is a library for parallel processing a numpy array using a pool of workers, each running on a separate process. It
performs the processing asynchronously so none of the work in starting the workers, or collecting their results when
finished, blocks.
Each worker handles processing a chunk of the array and MPAIO coordinates giving the results back to the user.
MPAIO expects the array to be processed to be available in shared memory and to remain constant.
MPAIO internally uses Python std library `ProcessPoolExecutor` to run the workers:
https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor-example
MPAIO uses anyio to do the asynchronous scheduling.

MPAIO is composed of:
    - a `DataIterator` class
        - encapsulates meta-data of the shared memory buffer and logic of how to divide the array amongst the workers.
    - an abstract `Worker` class
        - defines a template for a worker and the logic to process a chunk of the data.
        - a `Worker` takes the `DataIterator` object that it will be processing.
    - a `WorkerOrchestrator` class
        - runs the workers in the executor and handles the results.

MPAIO is designed using dependency injection, so the executor and shared memory must be created in the user code and
injected in when constructing the `WorkerOrchestrator`.

An example is included in `examples/` that sets up two shared memory arrays, one containing strings, the other
containing integers. For each of these arrays, a `Worker` is defined to process the data, finally each defines their own
`DataIterator` defining how the array should be batched.

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

Uses structured concurrency (anyio)
Use chatgpt for writing unit tests. This is a good litmus test for having small modular classes. ChatGPT generates
excellent tests for small, well designed simple classes whereas it struggles to test complex spaghetti classes.

Use a mix of anyio and asyncio
anyio - excellent library for structured concurrency, gives you task groups without having to be on Python 3.11. It does
not yet have support for synchronisation primitives for multi processing.
asyncio - run concurrent executor within asyncio.