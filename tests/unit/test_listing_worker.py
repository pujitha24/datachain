import asyncio
import io
import os
import pickle
import subprocess
import sys
from collections import namedtuple
from typing import ClassVar

import pytest

import datachain
from datachain.client import listing_worker
from datachain.client.fsspec import Client
from datachain.client.listing_worker import (
    Bound,
    Child,
    Pool,
    _cpu_quotas,
    _serve,
    dump,
    load,
    picklable,
    usable_cpus,
)

ROOT = os.path.dirname(os.path.dirname(datachain.__file__))


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
        (TwoArgsError(1, 2), TwoArgsError),
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


def test_child_reports_its_warm_up_error():
    out = io.BytesIO()

    kind, error = serve(FakeClient(out, fail=True), out)

    assert kind == "ready"
    assert isinstance(error, PermissionError)


def test_child_without_a_spec_exits_quietly():
    code = "from datachain.client.listing_worker import main; main()"
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        input=b"",
        capture_output=True,
        timeout=60,
        check=False,
        env={**os.environ, "PYTHONPATH": ROOT},
    )

    assert (out.returncode, out.stderr) == (0, b"")


def test_child_ignores_ctrl_c():
    code = (
        "import os, signal, sys\n"
        "from datachain.client import listing_worker\n"
        "listing_worker.load = lambda f: signal.raise_signal(signal.SIGINT)\n"
        "listing_worker.main()\n"
    )
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        input=b"",
        capture_output=True,
        timeout=60,
        check=False,
        env={**os.environ, "PYTHONPATH": ROOT},
    )

    assert (out.returncode, out.stderr) == (0, b"")


class StatusError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code
        self.exc_info = (None, None, (lambda: None))


class UnprintableError(Exception):
    def __init__(self):
        super().__init__()
        self.hook = lambda: None

    def __str__(self):
        raise ValueError("no message")


def test_errors_keep_the_attributes_that_pickle():
    sent = pickle.loads(pickle.dumps(picklable(StatusError("denied", 403))))  # noqa: S301

    assert type(sent) is StatusError
    assert (str(sent), sent.status_code, sent.exc_info) == ("denied", 403, None)


def test_errors_of_classes_that_do_not_pickle_keep_their_base_class():
    from botocore.exceptions import ClientError

    error_class = type("NoSuchBucket", (ClientError,), {})
    response = {"Error": {"Code": "NoSuchBucket"}, "ResponseMetadata": {}}
    error = error_class(response, "ListObjectsV2")

    sent = pickle.loads(pickle.dumps(picklable(error)))  # noqa: S301

    assert type(sent) is ClientError
    assert sent.response["Error"]["Code"] == "NoSuchBucket"


def test_an_error_that_neither_pickles_nor_prints_still_crosses():
    sent = pickle.loads(pickle.dumps(picklable(UnprintableError())))  # noqa: S301

    assert isinstance(sent, Exception)


@pytest.mark.parametrize(
    "files, expected",
    [
        ({"cpu.max": "max 100000"}, []),
        ({"cpu.max": "150000 100000"}, [(150000, 100000)]),
        (
            {"cpu.max": "max 100000", "user.slice/cpu.max": "200000 100000"},
            [(200000, 100000)],
        ),
    ],
)
def test_cpu_quotas_of_cgroup_v2_and_its_parents(files, expected, tmp_path):
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    (tmp_path / "cgroup").write_text("0::/user.slice\n")

    assert _cpu_quotas(str(tmp_path), str(tmp_path / "cgroup")) == expected


def test_cpu_quota_of_cgroup_v1(tmp_path):
    folder = tmp_path / "cpu,cpuacct" / "docker"
    folder.mkdir(parents=True)
    (folder / "cpu.cfs_quota_us").write_text("100000\n")
    (folder / "cpu.cfs_period_us").write_text("100000\n")
    (tmp_path / "cgroup").write_text("5:memory:/docker\n4:cpu,cpuacct:/docker\n")

    assert _cpu_quotas(str(tmp_path), str(tmp_path / "cgroup")) == [(100000, 100000)]


@pytest.mark.parametrize(
    "quotas, expected", [([], 16), ([(150000, 100000)], 1), ([(50000, 100000)], 1)]
)
def test_usable_cpus_follow_the_cgroup_quotas(quotas, expected, monkeypatch):
    monkeypatch.setattr(os, "process_cpu_count", lambda: 16, raising=False)
    monkeypatch.setattr(listing_worker, "_cpu_quotas", lambda: quotas)

    assert usable_cpus() == expected


def test_s3_connections_cover_the_listing_requests_in_flight():
    from datachain.client.fsspec import PROCESS_LIST_WORKERS
    from datachain.client.s3 import ClientS3

    fs = ClientS3.create_fs(anon=True, skip_instance_cache=True)
    mine = ClientS3.create_fs(
        anon=True, config_kwargs={"max_pool_connections": 5}, skip_instance_cache=True
    )

    assert fs.config_kwargs["max_pool_connections"] >= PROCESS_LIST_WORKERS
    assert mine.config_kwargs["max_pool_connections"] == 5
