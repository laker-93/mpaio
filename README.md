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