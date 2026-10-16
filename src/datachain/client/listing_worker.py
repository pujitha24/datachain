import asyncio
import copy
import logging
import os
import pickle
import signal
import struct
import sys
import threading
from collections.abc import AsyncGenerator
from contextlib import aclosing, suppress
from functools import partial
from typing import TYPE_CHECKING, Any, BinaryIO

if TYPE_CHECKING:
    from collections.abc import Callable

    from datachain.client.fsspec import Client

logger = logging.getLogger("datachain")


def _cpu_quotas(
    root: str = "/sys/fs/cgroup", membership: str = "/proc/self/cgroup"
) -> list[tuple[int, int]]:
    """``(quota, period)`` of each cgroup this process is in, and their parents."""
    quotas = []
    with suppress(OSError):
        with open(membership) as f:
            lines = f.read().splitlines()
        for line in lines:
            _, controllers, path = line.split(":", 2)
            if not controllers:  # v2
                base, files = root, ["cpu.max"]
            elif "cpu" in controllers.split(","):  # v1
                base = os.path.join(root, controllers)
                files = ["cpu.cfs_quota_us", "cpu.cfs_period_us"]
            else:
                continue
            parts = [p for p in path.split("/") if p]
            for depth in range(len(parts) + 1):
                folder = os.path.join(base, *parts[:depth])
                with suppress(OSError, ValueError):
                    quota, period = " ".join(
                        _read_text(os.path.join(folder, name)) for name in files
                    ).split()
                    if quota not in ("max", "-1"):
                        quotas.append((int(quota), int(period)))
    return quotas


def _read_text(path: str) -> str:
    with open(path) as f:
        return f.read()


def usable_cpus() -> int:
    """The CPUs this process may run on, within its cgroups' CPU quotas."""
    if hasattr(os, "process_cpu_count"):
        cpus = os.process_cpu_count() or 1
    elif hasattr(os, "sched_getaffinity"):
        cpus = len(os.sched_getaffinity(0))
    else:
        cpus = os.cpu_count() or 1
    for quota, period in _cpu_quotas():
        cpus = min(cpus, quota // period)
    return max(cpus, 1)


PROCESSES = 0 if usable_cpus() < 2 else min(4, usable_cpus())
FRAME = struct.Struct("<Q")
# files, first key, last key; files are ``row_type(File)`` rows
RangePage = tuple[list[Any], str | None, str | None]


class Bound:
    """The upper end of a range ``(lo, hi]``; it only shrinks, by a split. Keys
    compare in ``order``, the server's order."""

    def __init__(self, hi: str | None, order: "Callable[[str], Any]") -> None:
        self.order = order
        self.hi = hi
        self.on_shrink: Callable[[str], None] | None = None
        self.last: str | None = None  # the last key read so far
        self.first: str | None = None  # the first key of the last page
        self.pages = 0  # read by this range and the ranges it split off from
        self.probing = False  # a split waits for its probe
        self.done = False

    @property
    def hi(self) -> str | None:
        return self._hi

    @hi.setter
    def hi(self, hi: str | None) -> None:
        self._hi = hi
        self._hi_key = None if hi is None else self.order(hi)

    def shrink(self, hi: str) -> None:
        self.hi = hi
        if self.on_shrink:
            self.on_shrink(hi)

    def keeps(self, key: str) -> bool:
        return self._hi_key is None or self.order(key) <= self._hi_key


class ChildDied(Exception):  # noqa: N818
    pass


def item_path(item: Any) -> str:
    return item[1] if isinstance(item, tuple) else item.path  # path is column 1


def _pickles(value: Any) -> bool:
    try:
        pickle.dumps(value)
    except Exception:  # noqa: BLE001
        return False
    return True


def picklable(e: Exception) -> Exception:
    """``e``, or a copy of it that pickles: of its nearest class that pickles,
    with each attribute that does not pickle set to ``None``."""
    if _pickles(e):
        return e
    with suppress(Exception):
        err = copy.copy(e)
        if (cls := next(c for c in type(e).__mro__ if _pickles(c))) is not type(e):
            err.__class__ = cls
        for name, value in vars(err).items():
            if not _pickles(value):
                setattr(err, name, None)
        if _pickles(err):
            return err
    with suppress(Exception):
        return RuntimeError(f"{type(e).__name__}: {e}")
    return RuntimeError(type(e).__name__)


def dump(frame: tuple) -> bytes:
    data = pickle.dumps(frame, protocol=5)
    return FRAME.pack(len(data)) + data


def load(stream: BinaryIO) -> Any:
    header = stream.read(FRAME.size)
    if len(header) < FRAME.size:
        return None
    (size,) = FRAME.unpack(header)
    return pickle.loads(stream.read(size))  # noqa: S301


class Child:
    def __init__(self) -> None:
        self.proc: asyncio.subprocess.Process | None = None
        self.ranges: dict[int, asyncio.Queue] = {}
        self.died = asyncio.Event()

    async def start(self, spec: bytes) -> None:
        # the parent's imports, whatever the cwd holds
        code = f"import sys; sys.path[:] = {sys.path!r}; from {__name__} import main"
        self.proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            f"{code}; main()",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        self.send_raw(spec)
        kind, error = await self.read()
        if kind != "ready" or error is not None:
            raise ChildDied from error

    def send_raw(self, data: bytes) -> None:
        assert self.proc and self.proc.stdin
        if not self.died.is_set():
            self.proc.stdin.write(data)

    def send(self, *frame: Any) -> None:
        self.send_raw(dump(frame))

    async def read(self) -> Any:
        assert self.proc and self.proc.stdout
        (size,) = FRAME.unpack(await self.proc.stdout.readexactly(FRAME.size))
        return pickle.loads(await self.proc.stdout.readexactly(size))  # noqa: S301

    async def serve(self) -> None:
        try:
            while True:
                kind, rid, *rest = await self.read()
                if (queue := self.ranges.get(rid)) is not None:
                    await queue.put((kind, *rest))
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            self.died.set()

    async def get(self, queue: asyncio.Queue) -> tuple:
        if queue.empty() and self.died.is_set():
            raise ChildDied
        get = asyncio.ensure_future(queue.get())
        died = asyncio.ensure_future(self.died.wait())
        try:
            await asyncio.wait([get, died], return_when=asyncio.FIRST_COMPLETED)
        finally:
            died.cancel()
            if not get.done():
                get.cancel()
        if not get.done():
            raise ChildDied
        return get.result()

    def kill(self) -> None:
        self.died.set()
        if self.proc and self.proc.returncode is None:
            self.proc.kill()


class Pool:
    """Child processes listing ranges for one ``Client._fetch_ranges`` call."""

    def __init__(self, client: "Client", prefix: str, size: int = 0) -> None:
        try:
            self.spec = dump(
                (
                    type(client),
                    client.name,
                    client.fs_kwargs,
                    client.cache,
                    client.LIST_PAGE_SIZE,
                    prefix,
                )
            )
        except Exception:
            logger.debug(
                "Listing in one process: client config does not pickle", exc_info=True
            )
            size = 0
        self.size = size
        self.children: list[Child] = []
        self.tasks: list[asyncio.Task] = []
        self.next_id = 0
        self.closed = self.warned = False

    def start(self) -> None:
        if not self.tasks:
            self.tasks = [asyncio.create_task(self._run()) for _ in range(self.size)]

    async def _run(self) -> None:
        child = Child()
        try:
            await child.start(self.spec)
            self.children.append(child)
            await child.serve()
        except (
            OSError,
            EOFError,
            ChildDied,
            NotImplementedError,
            pickle.UnpicklingError,
        ):
            logger.debug("Listing process failed", exc_info=True)
        except Exception:
            logger.warning("Listing process failed", exc_info=True)
        finally:  # also when cancelled while the child starts
            if child in self.children:
                self.children.remove(child)
            child.kill()
            self._warn_when_alone()

    def _warn_when_alone(self) -> None:
        running = [t for t in self.tasks if not t.done()]  # this one is still running
        if not (self.closed or self.warned or self.children or len(running) > 1):
            self.warned = True
            logger.warning("Listing in one process: no listing process runs")

    def ready(self) -> bool:
        return bool(self.children)

    def usable(self) -> bool:
        """Whether a range may still move to a child process."""
        return bool(self.size) and (
            not self.tasks
            or bool(self.children)
            or not all(t.done() for t in self.tasks)
        )

    async def pages(
        self, prefix: str, lo: str, bound: Bound
    ) -> AsyncGenerator[RangePage, None]:
        from datachain.lib.convert.flatten import row_type
        from datachain.lib.file import File

        if not self.children:
            raise ChildDied
        file_row = partial(tuple.__new__, row_type(File))
        child = min(self.children, key=lambda c: len(c.ranges))
        rid = self.next_id = self.next_id + 1
        queue: asyncio.Queue = asyncio.Queue(2)
        child.ranges[rid] = queue
        child.send("open", rid, prefix, lo, bound.hi)
        bound.on_shrink = lambda hi: child.send("hi", rid, hi)
        try:
            while True:
                kind, *rest = await child.get(queue)
                if kind == "page":
                    files, first, last = rest
                    yield list(map(file_row, files)), first, last
                elif kind == "end":
                    return
                else:
                    raise rest[0]
        finally:
            bound.on_shrink = None
            del child.ranges[rid]
            child.send("close", rid)
            while not queue.empty():  # a reader blocked on this queue moves on
                queue.get_nowait()

    def close(self) -> None:
        self.closed = True
        for task in self.tasks:
            task.cancel()
        for child in self.children:
            child.kill()


async def _serve(  # noqa: C901
    client: "Client", prefix: str, inp: BinaryIO, out: BinaryIO
) -> None:
    loop = asyncio.get_running_loop()
    inbox: asyncio.Queue = asyncio.Queue()

    def pump() -> None:  # blocking stdin reads, also on Windows
        while (frame := load(inp)) is not None:
            loop.call_soon_threadsafe(inbox.put_nowait, frame)
        loop.call_soon_threadsafe(inbox.put_nowait, None)

    def write(*frame: Any) -> None:
        try:
            out.write(dump(frame))
            out.flush()
        except BrokenPipeError:  # the parent is gone
            os._exit(0)

    tasks: dict[int, asyncio.Task] = {}
    bounds: dict[int, Bound] = {}

    async def run(rid: int, prefix: str, lo: str) -> None:
        bound = bounds[rid]
        error: Exception | None = None
        try:
            pages = client._local_pages(prefix, lo, bound)
            async with aclosing(pages):
                async for files, first, last in pages:
                    # plain tuples: a row type would unpickle per row in Python
                    write("page", rid, [tuple(f) for f in files], first, last)
                    if last is not None and not bound.keeps(last):
                        break
            write("end", rid)
        except Exception as e:  # noqa: BLE001
            error = e
        finally:
            tasks.pop(rid, None)
            bounds.pop(rid, None)
        if error is not None:  # out of the handler: a copy takes no traceback
            write("error", rid, picklable(error))

    threading.Thread(target=pump, daemon=True).start()
    try:  # warm up DNS, TLS, credentials and the bucket region before taking work
        await client._probe(prefix, "")
    except Exception as e:  # noqa: BLE001
        write("ready", picklable(e))
        return
    write("ready", None)
    while (msg := await inbox.get()) is not None:
        kind, rid, *args = msg
        if kind == "open":
            prefix, lo, hi = args
            bounds[rid] = Bound(hi, client.key_order)
            tasks[rid] = asyncio.create_task(run(rid, prefix, lo))
        elif kind == "hi" and rid in bounds:
            bounds[rid].hi = args[0]
        elif kind == "close" and rid in tasks:
            tasks[rid].cancel()
    for task in list(tasks.values()):
        task.cancel()


def main() -> None:
    from fsspec.asyn import get_loop, sync

    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the parent closes stdin
    out = os.fdopen(os.dup(1), "wb")
    inp = os.fdopen(os.dup(0), "rb")
    os.dup2(2, 1)  # stray prints must not corrupt the frames
    with open(os.devnull, "rb") as devnull:  # nor a credential process read them
        os.dup2(devnull.fileno(), 0)
    if (spec := load(inp)) is None:
        return
    cls, name, fs_kwargs, cache, page_size, prefix = spec
    client = cls(name, fs_kwargs, cache)
    client.LIST_PAGE_SIZE = page_size
    sync(get_loop(), _serve, client, prefix, inp, out)


if __name__ == "__main__":
    main()
