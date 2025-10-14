@yawaramin
Concurrency is not about using CPU cores though. If you are talking about parallelism then Python has been able to use processes for a long time. I know that's not as 'efficient' as shared memory parallelism, but then again you're using Python so there was already a tradeoff made

@HeinrichKuttler
Python multiprocessing is overused garbage. You spend all your time serializing and deserializing pickled stuff. You have a "watchdog process" that attempts to detect if processes go away in the wrong order (memfd solves this correctly but is underused).

The _other_ issue with Python's multiprocessing is how it attempts to abstract away its implementation. Not even env vars can be set!

It probably does that in an attempt to have one interface across OSes, but it also just plain tries to do too much.

alternative: 

```python
import json
import socket
import os


class Worker:
    def __init__(self, work):
        sock, sock0 = socket.socketpair()
        pid = os.fork()

        if not pid:
            # Child. Could set PDEATHSIG here.
            sock0, sock = sock, sock0

        sock0.close()

        self.r = sock.makefile("r")
        self.w = sock.makefile("w")

        if not pid:
            work(self)
            os._exit(0)

        self.pid = pid

    def recv(self):
        # Could add timeouts with select.
        return json.loads(self.r.readline())

    def send(self, msg):
        json.dump(msg, self.w)
        self.w.write("\n")
        self.w.flush()

    def wait(self):
        _, status = os.waitpid(self.pid, 0)
        rc = os.waitstatus_to_exitcode(status)
        assert rc == 0


if __name__ == "__main__":
    # Example.

    def work(handle):
        rank, world_size = handle.recv()
        print(f"Hello from {rank}/{world_size}!")
        handle.send(2 * rank)

    num_workers = 2
    workers = [Worker(work) for _ in range(num_workers)]
    for i, w in enumerate(workers):
        w.send((i, len(workers)))

    for i, w in enumerate(workers):
        print(i, w.recv())

    for w in workers:
        w.wait()
```

This could also use marshal, and do shared memory with memfd_create, and potentially register different functions that are called via the simple interface.

And if you replace the UDS with a tcp socket you have the beginning of a distributed system here with much simpler semantics than, say, Ray, which also does too much.

elaboration on memfd approach:

You can do something like

fd = os.memfd_create(name)
os.ftruncate(fd, size)

and then either share fd with your child process e.g. via subprocess.Popen(pass_fds=) or you mmap it which multiprocessing can deserialize to the same region.

The kernel refcounts the fd like a file.

for locking writes: conceptually it's the same thing mp.Lock does, which is probably a mutex created by one side. I don't think this can be written "correctly" in C++ (or at least I don't know how), but it can be done in a way that will work on Linux. see facebookresearch/moolib


A related trick, on the more hacky side: If you _actually_ need a file path, you can still use that trick and then ask consumers to open /proc/self/fd/$fd. The 
@NetHack_LE
 uses this trick to re-open the same shared library w/o dlopen knowing it's the same file: 

@soumithchintala
pytorch uses this for sharing tensors across processes in torch.multiprocess

