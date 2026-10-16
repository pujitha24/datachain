import asyncio
import io
import pickle
from collections import namedtuple
from typing import ClassVar

import pytest

from datachain.client.fsspec import Client
from datachain.client.listing_worker import (
    Bound,
    Child,
    Pool,
    _serve,
    dump,
    load,
    picklable,
)


class FakeClient:
    def __init__(self, out, fail=False):
        self.out, self.fail, self.probed_before_ready = out, fail, None
        self.probed = None

    async def _probe(self, prefix, start_after):
        self.probed_before_ready = not self.out.getvalue()
        self.probed = prefix, start_after
        if self.fail:
            raise PermissionError("no list access")


def serve(client, out):
    inp = io.BytesIO()  # no requests: the child stops at end of input
    asyncio.run(_serve(client, "dir/", inp, out))
    out.seek(0)
    return load(out)


def test_child_warms_up_before_it_reports_ready():
    out = io.BytesIO()
    client = FakeClient(out)

    assert serve(client, out) == ("ready", None)
    assert client.probed_before_ready
    assert client.probed == ("dir/", "")


def test_child_reports_ready_when_warm_up_fails():
    out = io.BytesIO()

    assert serve(FakeClient(out, fail=True), out) == ("ready", None)


def test_frames_round_trip():
    assert load(io.BytesIO(dump(("page", 1, [("s3://b", "k")], "k", "k")))) == (
        "page",
        1,
        [("s3://b", "k")],
        "k",
        "k",
    )


def holding(error: Exception) -> Exception:
    error.handle = lambda: None  # type: ignore[attr-defined]
    return error


class TwoArgsError(Exception):
    def __init__(self, a, b):
        super().__init__(a, b)
        self.handle = lambda: None


@pytest.mark.parametrize(
    "error, sent_type",
    [
        (ValueError("x"), ValueError),
        (holding(OSError("x")), OSError),
        (TwoArgsError(1, 2), RuntimeError),
    ],
)
def test_errors_cross_the_pipe(error, sent_type):
    sent = pickle.loads(pickle.dumps(picklable(error)))  # noqa: S301

    assert type(sent) is sent_type
    assert str(error) in str(sent)


def test_azure_errors_cross_the_pipe():
    from azure.core.exceptions import ServiceRequestError

    try:
        try:
            raise OSError("connection reset")
        except OSError as e:
            raise ServiceRequestError("request failed") from e
    except ServiceRequestError as e:
        error = e

    sent = pickle.loads(pickle.dumps(picklable(error)))  # noqa: S301

    assert type(sent) is ServiceRequestError
    assert "request failed" in str(sent)


def test_pool_is_usable_until_every_child_is_gone():
    async def check():
        pool = Pool.__new__(Pool)
        pool.size, pool.tasks, pool.children = 0, [], []
        assert not pool.usable()

        pool.size = 2
        assert pool.usable()

        pool.tasks = [asyncio.create_task(asyncio.sleep(0))]
        assert pool.usable()
        await pool.tasks[0]
        assert not pool.usable()

    asyncio.run(check())


FileRow = namedtuple(  # noqa: PYI024
    "FileRow", "source path size version etag is_latest last_modified location"
)


def _file(path: str, version: str) -> FileRow:
    return FileRow("s3://bucket", path, 1, version, "", True, None, None)


class VersionsClient:
    name, cache, LIST_PAGE_SIZE = "bucket", None, 1
    fs_kwargs: ClassVar[dict] = {}
    _range_pages = Client._range_pages
    _local_pages = Client._local_pages

    def _lists_in_key_order(self):
        return True

    def _is_valid_key(self, key):
        return True

    key_order = staticmethod(Client.key_order)

    async def _pages_after(self, prefix, lo, keep, hi=None):
        for key, version in [("a", "v1"), ("a", "v2"), ("b", "v1")]:
            if key > lo:
                yield ([_file(key, version)] if keep(key) else []), key, key


def test_versions_survive_the_last_child_dying_mid_key(monkeypatch):
    async def check():
        opened = asyncio.Event()

        async def start(self, spec):
            pass

        def send(self, *frame):
            if frame[0] == "open":
                self.rid = frame[1]
                opened.set()

        async def read(self):
            await opened.wait()
            if not getattr(self, "sent", False):
                self.sent = True
                return ("page", self.rid, [tuple(_file("a", "v1"))], "a", "a")
            raise asyncio.IncompleteReadError(b"", 8)

        monkeypatch.setattr(Child, "start", start)
        monkeypatch.setattr(Child, "send", send)
        monkeypatch.setattr(Child, "read", read)
        client = VersionsClient()
        pool = Pool(client, "", 1)
        pool.start()
        await asyncio.sleep(0)

        async def versions(pool):
            pages = client._range_pages("", "", Bound(None, Client.key_order), pool)
            return [(f[1], f[3]) async for files, _, _ in pages for f in files]

        assert await versions(pool) == await versions(None)

    asyncio.run(check())
