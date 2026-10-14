import pickle
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Generic, TypeVar

import numpy as np
import pytest
import sqlalchemy as sa
from pydantic import BaseModel, Field

import datachain as dc
from datachain.lib.convert.flatten import flatten, iter_flat_columns
from datachain.query import dataset


class Info(BaseModel):
    a: int
    b: str = "x"


class Meta(BaseModel):
    name: str
    info: Info | None = None
    tags: list[str] = []


def test_row_type_fields_are_the_flat_columns():
    meta_row = dc.row_type(Meta)

    assert meta_row._fields == ("name", "info___type_tag", "info__a", "info__b", "tags")
    assert meta_row._field_defaults == {
        "info___type_tag": 1,
        "info__a": None,
        "info__b": None,
        "tags": (),
    }
    assert dc.row_type(dc.File)._fields == tuple(
        "__".join(c.path) for c in iter_flat_columns(dc.File)
    )
    assert dc.row_type(Meta) is meta_row


def test_row_type_rows_pickle():
    row = dc.row_type(dc.File)("s3://bucket", "a.txt")
    assert pickle.loads(pickle.dumps(row)) == row  # noqa: S301


def test_gen_rows_match_gen_models(test_session):
    meta_row = dc.row_type(Meta)

    def models(n: int) -> Iterator[Meta]:
        yield Meta(name=f"m{n}", info=Info(a=n), tags=["t"])
        yield Meta(name=f"e{n}")

    def rows(n: int) -> Iterator[Meta]:
        yield meta_row(f"m{n}", 0, n, "x", ["t"])
        yield meta_row(f"e{n}")

    chain = dc.read_values(n=[1, 2], session=test_session)
    expected = chain.gen(meta=models).order_by("meta.name").to_values("meta")

    assert chain.gen(meta=rows).order_by("meta.name").to_values("meta") == expected


def test_rows_in_multiple_and_optional_outputs(test_session):
    meta_row = dc.row_type(Meta)

    def fn(n: int) -> tuple[Meta | None, float]:
        return (meta_row(f"m{n}") if n else None), n / 2

    result = (
        dc.read_values(n=[0, 1], session=test_session)
        .map(fn, output={"meta": Meta | None, "score": float})
        .order_by("score")
        .to_list("meta", "score")
    )

    assert result == [(None, 0.0), (Meta(name="m1"), 0.5)]


def test_row_of_another_model_is_rejected(test_session):
    def info_row(n: int):
        return dc.row_type(Info)(n)

    chain = dc.read_values(n=[1], session=test_session).map(meta=info_row, output=Meta)

    with pytest.raises(TypeError, match="InfoRow"):
        chain.save("wrong")


def _stored(chain: dc.DataChain, session) -> list[tuple]:
    dataset = chain.dataset
    assert dataset is not None
    warehouse = session.catalog.warehouse
    table = warehouse.dataset_rows(dataset, dataset.latest_version).get_table()
    cols = [c for c in table.c if not c.name.startswith("sys__")]
    rows = warehouse.db.execute(sa.select(*cols))
    return sorted(((*r, *map(type, r)) for r in rows), key=repr)


def _files(n: int, as_rows: bool) -> Iterator[dc.File]:
    file_row = dc.row_type(dc.File)
    when = datetime(2026, 10, 13, 1, 2, 3, 456789, tzinfo=timezone.utc)
    for i in range(n):
        args = ("s3://bucket", f"{n}/{i}", i, "", f"e{i}", bool(i % 2), when)
        if as_rows and i != 1:
            yield file_row(*args)
        else:
            yield dc.File(**dict(zip(file_row._fields, args, strict=False)))


def test_gen_rows_are_stored_as_models(test_session):
    meta_row = dc.row_type(Meta)

    def models(n: int) -> Iterator[Meta]:
        yield from (Meta(name=f"m{n}", info=Info(a=n), tags=["t"]),) * n

    def rows(n: int) -> Iterator[Meta]:
        yield from (meta_row(f"m{n}", 0, n, "x", ["t"]),) * n

    chain = dc.read_values(n=[0, 1, 3], session=test_session)
    assert _stored(chain.gen(meta=rows).save("rows"), test_session) == _stored(
        chain.gen(meta=models).save("models"), test_session
    )

    files = chain.gen(file=lambda n: _files(n, True), output=dc.File).save("files")
    assert _stored(files, test_session) == _stored(
        chain.gen(file=lambda n: _files(n, False), output=dc.File).save("objects"),
        test_session,
    )


def test_gen_rows_skip_conversion(test_session, mocker):
    spy = mocker.spy(dataset, "adjust_outputs")

    dc.read_values(n=[3], session=test_session).gen(
        file=lambda n: _files(n, True), output=dc.File
    ).save("files")

    files = [c for c in spy.call_args_list if "file__path" in c.args[1]]
    assert len(files) == 1  # the File object among the rows


class Counted(BaseModel):
    x: int
    y: int = Field(default_factory=lambda data: data["x"] + 1)
    tags: list[str] = Field(default_factory=list)


def test_default_factories_run_per_model(test_session):
    def models(n: int) -> Iterator[Counted]:
        yield Counted(x=n)

    saved = dc.read_values(n=[1, 2], session=test_session).gen(c=models)

    assert dc.row_type(Counted)._field_defaults == {}
    assert sorted(saved.to_values("c"), key=lambda c: c.x) == [
        Counted(x=1, y=2),
        Counted(x=2, y=3),
    ]


class Bag(BaseModel):
    data: dict[str, set[int]]


def test_rows_with_json_values_are_stored_like_models(test_session):
    bag = Bag(data={"ids": {1, 2}})

    def models(n: int) -> Iterator[Bag]:
        yield bag

    def rows(n: int) -> Iterator[Bag]:
        yield dc.row_type(Bag)(*flatten(bag))

    chain = dc.read_values(n=[1], session=test_session)
    assert _stored(chain.gen(b=rows).save("rows"), test_session) == _stored(
        chain.gen(b=models).save("models"), test_session
    )


class Count(BaseModel):
    n: int


def test_numpy_values_are_stored_like_models(test_session):
    def models(n: int) -> Iterator[Count]:
        yield Count(n=12)

    def rows(n: int) -> Iterator[Count]:
        yield dc.row_type(Count)(np.uint64(12))

    chain = dc.read_values(n=[1], session=test_session)
    assert _stored(chain.gen(c=rows).save("rows"), test_session) == _stored(
        chain.gen(c=models).save("models"), test_session
    )


T = TypeVar("T")


class Box(BaseModel, Generic[T]):
    item: T


def test_generic_models_in_generators(test_session):
    def boxes(n: int) -> Iterator[Box[int]]:
        yield Box[int](item=n)

    chain = dc.read_values(n=[1, 2], session=test_session).gen(b=boxes)

    assert sorted(b.item for b in chain.to_values("b")) == [1, 2]


class Settings(BaseModel):
    name: str = "s"
    values: dict[str, int] = {}


class Nested(BaseModel):
    values: list[list[int]] = [[1]]


class Items(BaseModel):
    items: list[Info] = [Info(a=1)]


def test_row_defaults_are_immutable():
    with pytest.raises(AttributeError):
        dc.row_type(Meta)("m").tags.append("t")  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        dc.row_type(Nested)().values[0].append(2)  # type: ignore[attr-defined]

    assert Meta(name="m").tags == []
    assert dc.row_type(Settings)._field_defaults == {}
    assert dc.row_type(Items)._field_defaults == {}


class Either(BaseModel):
    value: bool | Info


def test_json_columns_are_stored_like_models(test_session):
    def models(n: int) -> Iterator[Either]:
        yield Either(value=True)

    def rows(n: int) -> Iterator[Either]:
        yield dc.row_type(Either)(True)

    chain = dc.read_values(n=[1], session=test_session)
    assert _stored(chain.gen(e=rows).save("rows"), test_session) == _stored(
        chain.gen(e=models).save("models"), test_session
    )


class Outer(BaseModel):
    inner: Info = Field(default={"a": 1}, validate_default=True)


def test_validated_defaults_in_generators(test_session):
    def outers(n: int) -> Iterator[Outer]:
        yield Outer()

    chain = dc.read_values(n=[1], session=test_session).gen(o=outers)

    assert chain.to_values("o") == [Outer()]


def test_rows_of_models_with_validated_defaults(test_session):
    outer_row = dc.row_type(Outer)

    def rows(n: int) -> Iterator[Outer]:
        yield outer_row(1, "x")

    chain = dc.read_values(n=[1], session=test_session).gen(o=rows)

    assert outer_row._field_defaults == {}
    assert chain.to_values("o") == [Outer(inner=Info(a=1, b="x"))]


class Label(BaseModel):
    text: str


def test_rows_go_through_an_overridden_convert_type(test_session, monkeypatch):
    def convert_type(self, val, *args):
        return val.upper() if isinstance(val, str) else val

    monkeypatch.setattr(
        type(test_session.catalog.warehouse), "convert_type", convert_type
    )
    dataset._keeps_exact_types.cache_clear()

    def models(n: int) -> Iterator[Label]:
        yield Label(text="hello")

    def rows(n: int) -> Iterator[Label]:
        yield dc.row_type(Label)("hello")

    chain = dc.read_values(n=[1], session=test_session)
    try:
        stored = _stored(chain.gen(t=rows).save("rows"), test_session)
        assert stored == _stored(chain.gen(t=models).save("models"), test_session)
        assert stored[0][0] == "HELLO"
    finally:
        dataset._keeps_exact_types.cache_clear()
