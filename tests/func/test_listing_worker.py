import asyncio
import time
from contextlib import aclosing

import pytest
from fsspec.asyn import sync

import datachain as dc
from datachain.asyn import get_loop
from datachain.client import Client
from datachain.client.listing_worker import Child, Pool, item_path
from datachain.lib.convert.flatten import flatten
from datachain.lib.file import File, FileRow
from datachain.lib.listing import list_bucket

TREE = {
    "a": "1",
    "big": {f"{i:03}": "x" for i in range(60)},
    "flat": {f"{i * 7919 % 4096:03x}": "x" for i in range(30)},
    "z": "1",
}


@pytest.fixture
def client(cloud_server, cloud_server_credentials, mocker):
    uri = cloud_server.src_uri
    client = Client.get_implementation(uri).from_source(
        uri, cache=None, **cloud_server.client_config
    )
    mocker.patch.object(type(client), "LIST_PAGE_SIZE", 4)
    mocker.patch("datachain.client.listing_worker.PROCESSES", 2)
    return client


@pytest.fixture
def children(client, mocker):
    """Children of the listing; once they are starting, the parent lists no
    further page until one runs, so every listing past two pages moves to them."""
    started: list[Pool] = []
    pool_start, local_pages = Pool.start, type(client)._local_pages

    def start(self):
        started.append(self)
        pool_start(self)

    async def gated(self, prefix, lo, bound, limit=None):
        async for page in local_pages(self, prefix, lo, bound, limit):
            deadline = time.monotonic() + 30
            while starting(started) and time.monotonic() < deadline:  # noqa: ASYNC110
                await asyncio.sleep(0.01)
            yield page

    mocker.patch.object(Pool, "start", start)
    mocker.patch.object(type(client), "_local_pages", gated)
    return started


@pytest.fixture
def remote_rows(mocker):
    got: list = []
    pages = Pool.pages

    async def recorded(self, prefix, lo, bound):
        async for page in pages(self, prefix, lo, bound):
            got.extend(page[0])
            yield page

    mocker.patch.object(Pool, "pages", recorded)
    return got


def starting(pools):
    pool = pools[-1] if pools else None
    return pool and not pool.ready() and any(not t.done() for t in pool.tasks)


async def _items(client, prefix, processes):
    items = []
    pages = client.scandir(prefix, processes=processes, rows=True)
    async with aclosing(pages) as pages:
        async for page in pages:
            items.extend(page)
    return items


def listing(client, prefix="", processes=True):
    return rows(sync(get_loop(), _items, client, prefix, processes))


def rows(items):
    return sorted(tuple(i) if isinstance(i, tuple) else flatten(i) for i in items)


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
@pytest.mark.parametrize("tree", [TREE], indirect=True)
@pytest.mark.parametrize("prefix", ["", "big", "flat"])
def test_listing_in_children_is_exact(client, tree, prefix, children, remote_rows):
    items = sync(get_loop(), _items, client, prefix, True)

    assert remote_rows
    assert rows(items) == listing(client, prefix, processes=False)


# fake-gcs-server repeats a page forever when one key's versions span pages
@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
@pytest.mark.parametrize("version_aware", [True], indirect=True)
@pytest.mark.parametrize("tree", [TREE], indirect=True)
def test_versions_across_pages_are_exact(client, tree, children, remote_rows):
    for i in range(9):
        client.fs.pipe_file(f"{client.uri}/big/030", str(i).encode())

    items = sync(get_loop(), _items, client, "", True)

    assert remote_rows
    assert rows(items) == listing(client, processes=False)
    assert sum(item_path(i) == "big/030" for i in items) == 10


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
@pytest.mark.parametrize("tree", [TREE], indirect=True)
@pytest.mark.parametrize("processes", [1, 2])
def test_ranges_of_a_dead_child_move_on(client, tree, processes, children, mocker):
    mocker.patch("datachain.client.listing_worker.PROCESSES", processes)
    killed = []
    read = Child.read

    async def kill_after_a_page(self):
        frame = await read(self)
        if frame[0] == "page" and not killed:
            killed.append(self)
            self.proc.kill()
        return frame

    mocker.patch.object(Child, "read", kill_after_a_page)

    assert listing(client) == listing(client, processes=False)
    assert killed


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
@pytest.mark.parametrize("tree", [TREE], indirect=True)
def test_children_that_never_start_leave_the_listing_exact(client, tree, mocker):
    mocker.patch.object(Child, "start", side_effect=OSError("no process"))

    assert listing(client) == listing(client, processes=False)


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
@pytest.mark.parametrize("tree", [TREE], indirect=True)
def test_children_stop_when_listing_is_closed_early(
    client, tree, children, remote_rows, mocker
):
    spawned = []
    child_start = Child.start

    async def start(self, spec):
        spawned.append(self)
        await child_start(self, spec)

    mocker.patch.object(Child, "start", start)

    async def first_child_page():
        async with aclosing(client.scandir("", processes=True)) as pages:
            async for _ in pages:
                if remote_rows:
                    return

    sync(get_loop(), first_child_page)
    procs = [c.proc for c in spawned if c.proc]
    deadline = time.monotonic() + 10
    while any(p.returncode is None for p in procs) and time.monotonic() < deadline:
        time.sleep(0.05)

    assert procs
    assert all(p.returncode is not None for p in procs)


@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
@pytest.mark.parametrize("tree", [TREE], indirect=True)
def test_config_that_does_not_pickle_lists_in_process(client, tree, remote_rows):
    from aiobotocore.session import AioSession

    client.fs_kwargs = {**client.fs_kwargs, "session": AioSession()}

    assert Pool(client, "", 2).size == 0
    items = sync(get_loop(), _items, client, "", True)
    assert items
    assert not remote_rows


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
@pytest.mark.parametrize("tree", [TREE], indirect=True)
def test_read_storage_rows_from_children(client, tree, children):
    uri = f"{client.uri}/"
    chain = dc.read_values(seed=[0]).gen(
        file=list_bucket(uri, None, client_config=client.fs_kwargs)
    )

    files = chain.order_by("file.path", "file.version").to_values("file")

    assert children
    assert sorted(flatten(f) for f in files) == listing(client, processes=False)


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
@pytest.mark.parametrize("tree", [TREE], indirect=True)
def test_listing_rows_are_normalized_files(client, tree):
    rows = sync(get_loop(), _items, client, "", False)

    assert rows
    assert all(type(r) is FileRow for r in rows)
    assert [flatten(File(**r._asdict())) for r in rows] == [tuple(r) for r in rows]


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
@pytest.mark.parametrize("tree", [TREE], indirect=True)
@pytest.mark.parametrize(
    "file_type", [dc.ImageFile, dc.TextFile, dc.VideoFile, dc.AudioFile]
)
def test_file_types_from_children(
    client, tree, children, remote_rows, test_session, file_type
):
    files = (
        dc.read_values(seed=[0], session=test_session)
        .gen(
            file=list_bucket(f"{client.uri}/", None, client_config=client.fs_kwargs),
            output={"file": file_type},
        )
        .to_values("file")
    )

    assert remote_rows
    assert {type(f) for f in files} == {file_type}
    assert sorted(flatten(f) for f in files) == listing(client, processes=False)


@pytest.fixture
def requests_in_flight(client, mocker):
    """The most page requests the parent had open at once."""
    peak = [0]
    now = 0
    pages_after = type(client)._pages_after

    async def counted(self, *args, **kwargs):
        nonlocal now
        pages = pages_after(self, *args, **kwargs)
        async with aclosing(pages):
            while True:
                now += 1
                peak[0] = max(peak[0], now)
                try:
                    page = await anext(pages)
                except StopAsyncIteration:
                    return
                finally:
                    now -= 1
                yield page

    mocker.patch.object(type(client), "_pages_after", counted)
    return peak


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
@pytest.mark.parametrize("tree", [{f"{i:03}": "x" for i in range(400)}], indirect=True)
def test_parent_lists_with_its_own_budget_when_no_child_runs(
    client, tree, requests_in_flight, mocker, caplog
):
    mocker.patch("datachain.client.fsspec.LIST_WORKERS", 3)
    mocker.patch.object(Child, "start", side_effect=OSError("no process"))

    with caplog.at_level("WARNING", logger="datachain"):
        assert len(listing(client)) == len(tree)

    assert requests_in_flight[0] <= 3
    warnings = [
        r for r in caplog.records if r.name == "datachain" and r.levelname == "WARNING"
    ]
    assert [r.getMessage() for r in warnings] == [
        "Listing in one process: no listing process runs"
    ]


@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
@pytest.mark.parametrize("tree", [TREE], indirect=True)
def test_children_import_datachain_whatever_the_cwd_holds(
    client, tree, children, remote_rows, tmp_path, monkeypatch
):
    (tmp_path / "fsspec.py").write_text("raise ImportError('shadowed')\n")
    monkeypatch.chdir(tmp_path)

    assert listing(client) == listing(client, processes=False)
    assert remote_rows


CREDENTIAL_PROCESS = """
import json, os, stat, sys
with open(sys.argv[1], "a") as log:
    log.write(f"{os.getppid()} {stat.S_ISFIFO(os.fstat(0).st_mode)}\\n")
print(json.dumps({"Version": 1, "AccessKeyId": sys.argv[2],
                  "SecretAccessKey": sys.argv[3], "SessionToken": sys.argv[4]}))
"""


@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
@pytest.mark.parametrize("tree", [TREE], indirect=True)
def test_children_take_credentials_from_a_credential_process(
    client, tree, children, remote_rows, tmp_path, monkeypatch
):
    import os
    import sys

    script, log = tmp_path / "creds.py", tmp_path / "creds.log"
    script.write_text(CREDENTIAL_PROCESS)
    kwargs = dict(client.fs_kwargs)
    keys = [kwargs.pop(k) for k in ("aws_key", "aws_secret", "aws_token")]
    config = tmp_path / "config"
    config.write_text(
        "[profile listing]\n"
        f"credential_process = {sys.executable} {script} {log} {' '.join(keys)}\n"
    )
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
    for name in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"]:
        monkeypatch.delenv(name, raising=False)
    client.fs_kwargs = {**kwargs, "profile": "listing"}
    client._fs = None

    assert len(listing(client)) == len(listing(client, processes=False))
    assert remote_rows
    calls = [line.split() for line in log.read_text().splitlines()]
    in_children = [fifo for ppid, fifo in calls if int(ppid) != os.getpid()]
    assert in_children
    assert set(in_children) == {"False"}  # the frame pipe is not their stdin
