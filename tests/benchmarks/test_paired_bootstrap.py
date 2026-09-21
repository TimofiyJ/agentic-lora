import json

from benchmarks.paired_bootstrap import (
    compare_domain,
    load_scores_by_task_id,
    paired_bootstrap,
)


def test_paired_bootstrap_returns_expected_keys():
    result = paired_bootstrap([0.0] * 20, [0.5] * 20, n_resamples=100, seed=0)
    assert set(result) == {"mean_diff", "ci_low", "ci_high", "p_value"}


def test_paired_bootstrap_detects_clear_improvement():
    scores_a = [0.0] * 20  # B0
    scores_b = [1.0] * 20  # OURS, always better
    result = paired_bootstrap(scores_a, scores_b, n_resamples=1000, seed=0)
    assert result["mean_diff"] == 1.0
    assert result["ci_low"] > 0
    assert result["p_value"] == 0.0


def test_paired_bootstrap_detects_no_difference():
    scores_a = [0.5] * 20
    scores_b = [0.5] * 20
    result = paired_bootstrap(scores_a, scores_b, n_resamples=1000, seed=0)
    assert result["mean_diff"] == 0.0
    assert result["p_value"] == 1.0  # every resample's mean diff is exactly 0, never > 0


def test_paired_bootstrap_is_deterministic_for_a_fixed_seed():
    scores_a = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    scores_b = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    a = paired_bootstrap(scores_a, scores_b, n_resamples=500, seed=42)
    b = paired_bootstrap(scores_a, scores_b, n_resamples=500, seed=42)
    assert a == b


def test_load_scores_by_task_id_extracts_domain_scores(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({
        "tasks": [
            {"id": "quality_sql_01", "domain": "sql", "score": 0.5},
            {"id": "quality_sql_02", "domain": "sql", "score": 1.0},
            {"id": "quality_legal_01", "domain": "legal", "score": 0.3},
        ]
    }))
    scores = load_scores_by_task_id(path, "sql")
    assert scores == {"quality_sql_01": 0.5, "quality_sql_02": 1.0}


def test_compare_domain_matches_by_task_id_not_list_position(tmp_path):
    """The two result files may list tasks in different orders -- pairing
    must be by task id, not by position in the tasks list."""
    b0_path = tmp_path / "b0.json"
    ours_path = tmp_path / "ours.json"
    b0_path.write_text(json.dumps({
        "tasks": [
            {"id": "quality_sql_01", "domain": "sql", "score": 0.0},
            {"id": "quality_sql_02", "domain": "sql", "score": 0.0},
        ]
    }))
    # Deliberately reversed order relative to b0.json.
    ours_path.write_text(json.dumps({
        "tasks": [
            {"id": "quality_sql_02", "domain": "sql", "score": 1.0},
            {"id": "quality_sql_01", "domain": "sql", "score": 0.0},
        ]
    }))

    result = compare_domain(b0_path, ours_path, "sql", n_resamples=100, seed=0)

    assert result["n_paired_tasks"] == 2
    assert result["mean_diff"] == 0.5  # (0.0 -> 0.0) and (0.0 -> 1.0), mean improvement 0.5
