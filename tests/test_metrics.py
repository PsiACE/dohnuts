import math

import pytest

from dohnuts.metrics import fit_temperatures, summarize


def test_twenty_latency_samples_do_not_report_maximum_as_p95():
    from dohnuts.experiment import latency_stats

    result = latency_stats(list(range(1, 21)), 1)
    assert result["p95_ms"] == pytest.approx(19.05, rel=0, abs=5e-8)


def test_known_binary_distribution():
    rows = [
        {"type": "noul", "logits": [0.0, math.log(4)], "target": t}
        for t in [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
    ]
    result = summarize(rows)
    assert result["accuracy"] == pytest.approx(0.75, rel=0, abs=5e-8)
    assert result["ece_15"] == pytest.approx(0.05, rel=0, abs=5e-8)
    expected_nll = -(0.75 * math.log(0.8) + 0.25 * math.log(0.2))
    assert result["nll"] == pytest.approx(expected_nll, rel=0, abs=5e-8)
    assert result["brier_sum"] == pytest.approx(0.38, rel=0, abs=5e-8)
    assert result["brier_per_candidate"] == pytest.approx(0.19, rel=0, abs=5e-8)
    empty_bins = [row for row in result["reliability"] if row["n"] == 0]
    assert len(empty_bins) == 14
    for row in empty_bins:
        assert row["confidence"] is None
        assert row["accuracy"] is None


def test_temperature_fits_soft_targets_without_changing_order():
    rows = [{"type": "noul", "logits": [0.0, 4.0], "target": [0.25, 0.75]} for _ in range(20)]
    temperatures = fit_temperatures(rows)
    assert temperatures["noul"] == pytest.approx(4 / math.log(3), rel=0, abs=0.02)
    assert summarize(rows, temperatures)["nll"] < summarize(rows)["nll"]


def test_ordinal_distance_and_soft_accuracy():
    result = summarize(
        [{"type": "score", "logits": [-100.0, -100.0, 0.0], "target": [1.0, 0.0, 0.0]}]
    )
    assert result["rps"] == pytest.approx(1.0, rel=0, abs=5e-8)
    assert result["score_mae"] == pytest.approx(2.0, rel=0, abs=5e-8)
    assert result["soft_accuracy"] == 0.0
