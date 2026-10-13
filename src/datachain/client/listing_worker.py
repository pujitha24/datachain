import asyncio
import logging
import os
import pickle
import struct
import sys
import threading
from collections.abc import AsyncGenerator
from contextlib import aclosing, suppress
from typing import TYPE_CHECKING, Any, BinaryIO

if TYPE_CHECKING:
    from collections.abc import Callable

    from datachain.client.fsspec import Client

logger = logging.getLogger("datachain")

PROCESSES = min(4, (os.cpu_count() or 1) - 1)
FRAME = struct.Struct("<Q")
# files, first key, last key; files are ``File`` objects or flat ``File`` rows
RangePage = tuple[list[Any], str | None, str | None]


class Bound:
    """The upper end of a range ``(lo, hi]``; it only shrinks, by a split."""

    def __init__(self, hi: str | None) -> None:
        self.hi = hi
        self.on_shrink: Callable[[str], None] | None = None
        self.last: str | None = None  # the last key read so far
        self.first: str | None = None  # the first key of the last page
        self.pages = 0  # read by this range and the ranges it split off from
        self.probing = False  # a split waits for its probe
        self.done = False

    def shrink(self, hi: str) -> None:
        self.hi = hi
        if self.on_shrink:
            self.on_shrink(hi)

    def keeps(self, key: str) -> bool:
        return self.hi is None or key <= self.hi


class ChildDied(Exception):  # noqa: N818
    pass


def item_path(item: Any) -> str:
    return item[1] if isinstance(item, tuple) else item.path  # path is column 1


def picklable(e: Exception) -> Exception:
    """``e``, or an exception of its type and message when ``e`` holds what does
    not pickle, such as a traceback."""
    with suppress(Exception):
        pickle.dumps(e)
        return e
    with suppress(Exception):
        err = type(e)(str(e))
        pickle.dumps(err)
        return err
    return RuntimeError(f"{type(e).__name__}: {e}")


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
        # the child must import this datachain, whatever the cwd or PYTHONPATH
        root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        path = os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")]))
        self.proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            __name__,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            env={**os.environ, "PYTHONPATH": path},
        )
        self.send_raw(spec)
        if await self.read() != ("ready", None):
            raise ChildDied

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

    def __init__(self, client: "Client", size: int = 0) -> None:
        try:
            self.spec = dump(
                (
                    type(client),
                    client.name,
                    client.fs_kwargs,
                    client.cache,
                    client.LIST_PAGE_SIZE,
                )
            )
        except Exception:  # noqa: BLE001
            logger.debug("Listing in one process: client config does not pickle")
            size = 0
        self.size = size
        self.children: list[Child] = []
        self.tasks: list[asyncio.Task] = []
        self.next_id = 0

    def start(self) -> None:
        if not self.tasks:
            self.tasks = [asyncio.create_task(self._run()) for _ in range(self.size)]

    async def _run(self) -> None:
        child = Child()
        try:
            await child.start(self.spec)
            self.children.append(child)
            await child.serve()
        except Exception:
            logger.debug("Listing process failed", exc_info=True)
        finally:  # also when cancelled while the child starts
            if child in self.children:
                self.children.remove(child)
            child.kill()

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
        if not self.children:
            raise ChildDied
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
                    yield tuple(rest)  # type: ignore[misc]
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
        for task in self.tasks:
            task.cancel()
        for child in self.children:
            child.kill()


async def _serve(client: "Client", inp: BinaryIO, out: BinaryIO) -> None:
    from datachain.lib.convert.flatten import flatten

    loop = asyncio.get_running_loop()
    inbox: asyncio.Queue = asyncio.Queue()

    def pump() -> None:  # blocking stdin reads, also on Windows
        while (frame := load(inp)) is not None:
            loop.call_soon_threadsafe(inbox.put_nowait, frame)
        loop.call_soon_threadsafe(inbox.put_nowait, None)

    def write(*frame: Any) -> None:
        out.write(dump(frame))
        out.flush()

    tasks: dict[int, asyncio.Task] = {}
    bounds: dict[int, Bound] = {}

    async def run(rid: int, prefix: str, lo: str) -> None:
        bound = bounds[rid]
        error: Exception | None = None
        try:
            pages = client._local_pages(prefix, lo, bound)
            async with aclosing(pages):
                async for files, first, last in pages:
                    write("page", rid, [flatten(f) for f in files], first, last)
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
        await client._probe("", "")
    except Exception:
        logger.debug("Listing process warm-up failed", exc_info=True)
    write("ready", None)
    while (msg := await inbox.get()) is not None:
        kind, rid, *args = msg
        if kind == "open":
            prefix, lo, hi = args
            bounds[rid] = Bound(hi)
            tasks[rid] = asyncio.create_task(run(rid, prefix, lo))
        elif kind == "hi" and rid in bounds:
            bounds[rid].hi = args[0]
        elif kind == "close" and rid in tasks:
            tasks[rid].cancel()
    for task in list(tasks.values()):
        task.cancel()


def main() -> None:
    from fsspec.asyn import get_loop, sync

    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)  # stray prints must not corrupt the frames
    cls, name, fs_kwargs, cache, page_size = load(sys.stdin.buffer)
    client = cls(name, fs_kwargs, cache)
    client.LIST_PAGE_SIZE = page_size
    sync(get_loop(), _serve, client, sys.stdin.buffer, out)


if __name__ == "__main__":
    main()
