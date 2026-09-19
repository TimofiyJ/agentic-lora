from training.data.sources import SOURCES


def test_sources_cover_all_three_p0_domains():
    assert set(SOURCES) == {"sql", "legal", "medical"}


def test_every_source_has_a_40_char_hex_revision():
    for domain, source in SOURCES.items():
        assert len(source.revision) == 40, domain
        int(source.revision, 16)  # raises ValueError if not hex


def test_legal_source_has_no_exclusion_since_it_uses_a_different_split_than_the_benchmark():
    assert SOURCES["legal"].exclude_head_n == 0


def test_sql_and_medical_sources_exclude_the_benchmarks_head_of_stream():
    assert SOURCES["sql"].exclude_head_n == 200
    assert SOURCES["medical"].exclude_head_n == 200


def test_legal_uses_train_split_while_benchmark_uses_test():
    assert SOURCES["legal"].split == "train"
