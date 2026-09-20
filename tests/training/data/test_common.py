import json

from training.data import common
from training.data.sources import DomainSource


def test_skip_head_skips_first_n_items():
    assert list(common.skip_head([1, 2, 3, 4, 5], 2)) == [3, 4, 5]


def test_skip_head_with_n_larger_than_input_yields_nothing():
    assert list(common.skip_head([1, 2], 5)) == []


def test_skip_head_with_zero_yields_everything():
    assert list(common.skip_head([1, 2, 3], 0)) == [1, 2, 3]


def test_format_rows_drops_rows_that_format_row_rejects():
    def format_row(row):
        return {"messages": []} if row["keep"] else None

    result = common.format_rows(
        [{"keep": True}, {"keep": False}, {"keep": True}], format_row
    )
    assert len(result) == 2


def test_train_val_split_is_deterministic_for_a_fixed_seed():
    examples = [{"i": i} for i in range(10)]
    train_a, val_a = common.train_val_split(examples, val_size=3, seed=42)
    train_b, val_b = common.train_val_split(examples, val_size=3, seed=42)
    assert train_a == train_b
    assert val_a == val_b
    assert len(val_a) == 3
    assert len(train_a) == 7


def test_train_val_split_does_not_overlap_and_covers_everything():
    examples = [{"i": i} for i in range(10)]
    train, val = common.train_val_split(examples, val_size=3, seed=1)
    train_ids = {e["i"] for e in train}
    val_ids = {e["i"] for e in val}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == set(range(10))


def test_write_jsonl_round_trips(tmp_path):
    path = tmp_path / "out.jsonl"
    examples = [{"a": 1}, {"a": 2}]
    common.write_jsonl(path, examples)
    lines = path.read_text().splitlines()
    assert [json.loads(line) for line in lines] == examples


def test_write_manifest_contains_source_and_counts(tmp_path):
    path = tmp_path / "manifest.json"
    source = DomainSource(
        repo_id="org/repo",
        config=None,
        split="train",
        revision="a" * 40,
        exclude_head_n=200,
        required_fields=("x",),
    )
    common.write_manifest(
        path, "sql", source,
        n_rows_before_filter=10, n_rows_after_filter=8, n_train=6, n_val=2,
    )
    manifest = json.loads(path.read_text())
    assert manifest["domain"] == "sql"
    assert manifest["repo_id"] == "org/repo"
    assert manifest["revision"] == "a" * 40
    assert manifest["exclude_head_n"] == 200
    assert manifest["n_train"] == 6
    assert manifest["n_val"] == 2
    assert "generated_at" in manifest


def test_stream_valid_rows_filters_out_rows_missing_required_fields(monkeypatch):
    fake_rows = [
        {"a": "1", "b": "1"},
        {"a": "", "b": "1"},
        {"a": "2", "b": None},
        {"a": "3", "b": "3"},
    ]

    class FakeDataset:
        def __iter__(self):
            return iter(fake_rows)

    monkeypatch.setattr(
        common.datasets, "load_dataset", lambda *args, **kwargs: FakeDataset()
    )

    source = DomainSource(
        repo_id="org/repo",
        config=None,
        split="train",
        revision="a" * 40,
        exclude_head_n=0,
        required_fields=("a", "b"),
    )
    result = list(common.stream_valid_rows(source))
    assert result == [{"a": "1", "b": "1"}, {"a": "3", "b": "3"}]


def test_run_prepare_writes_train_val_and_manifest(tmp_path, monkeypatch):
    fake_rows = [{"x": str(i)} for i in range(5)]
    monkeypatch.setattr(common, "stream_valid_rows", lambda source: iter(fake_rows))
    # SOURCES["sql"].exclude_head_n is 200 in production config; skip_head is
    # unit-tested on its own above, so bypass it here to isolate run_prepare's
    # own split/write wiring from that real-world constant.
    monkeypatch.setattr(common, "skip_head", lambda rows, n: rows)

    def format_row(row):
        return {
            "messages": [
                {"role": "user", "content": row["x"]},
                {"role": "assistant", "content": "y"},
            ]
        }

    common.run_prepare("sql", format_row, out_dir=str(tmp_path), val_size=2)

    train = [json.loads(l) for l in (tmp_path / "train.jsonl").read_text().splitlines()]
    val = [json.loads(l) for l in (tmp_path / "val.jsonl").read_text().splitlines()]
    manifest = json.loads((tmp_path / "manifest_sql.json").read_text())

    assert len(train) == 3
    assert len(val) == 2
    assert manifest["n_train"] == 3
    assert manifest["n_val"] == 2
    assert manifest["n_rows_before_filter"] == 5
    assert manifest["n_rows_after_filter"] == 5


def test_subsample_returns_exactly_n_items():
    examples = [{"i": i} for i in range(100)]
    result = common.subsample(examples, n=10, seed=0)
    assert len(result) == 10


def test_subsample_is_deterministic_for_a_fixed_seed():
    examples = [{"i": i} for i in range(100)]
    a = common.subsample(examples, n=10, seed=0)
    b = common.subsample(examples, n=10, seed=0)
    assert a == b


def test_subsample_items_are_all_from_the_original_list():
    examples = [{"i": i} for i in range(100)]
    result = common.subsample(examples, n=10, seed=0)
    original_ids = {e["i"] for e in examples}
    assert all(e["i"] in original_ids for e in result)


def test_batch_splits_into_chunks_of_the_given_size():
    items = list(range(10))
    result = list(common.batch(items, 3))
    assert result == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


def test_batch_with_size_larger_than_input_yields_one_batch():
    items = list(range(3))
    result = list(common.batch(items, 10))
    assert result == [[0, 1, 2]]
