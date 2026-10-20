import numpy as np
import pytest

from varmc.backtest import basel_zone, christoffersen_independence, kupiec_pof


def test_kupiec_accepts_expected_rate():
    lr, p = kupiec_pof(10, 1000, 0.01)
    assert lr == pytest.approx(0.0, abs=1e-9)
    assert p == pytest.approx(1.0)


def test_kupiec_rejects_too_many_exceptions():
    _, p = kupiec_pof(30, 1000, 0.01)
    assert p < 0.001


def test_kupiec_handles_zero_exceptions():
    lr, p = kupiec_pof(0, 1000, 0.01)
    assert lr == pytest.approx(-2 * 1000 * np.log(0.99))
    assert 0 < p < 0.01


def test_christoffersen_detects_clustering():
    clustered = np.zeros(1000, dtype=bool)
    clustered[500:510] = True                       # dix exceptions consécutives
    _, p_clustered = christoffersen_independence(clustered)
    spread = np.zeros(1000, dtype=bool)
    spread[::100] = True                            # dix exceptions isolées
    _, p_spread = christoffersen_independence(spread)
    assert p_clustered < 0.001
    assert p_spread > 0.5


@pytest.mark.parametrize("n, zone", [(0, "vert"), (4, "vert"), (5, "orange"), (9, "orange"), (10, "rouge")])
def test_basel_zone(n, zone):
    assert basel_zone(n) == zone
