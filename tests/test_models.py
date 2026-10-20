import numpy as np
import pandas as pd
import pytest

from varmc.data import eur_log_returns
from varmc.models import (ewma_covariances, fit_student_dof, gaussian_risk, historical_risk,
                          monte_carlo_risk, risk_from_pnl, simulate_returns)

COV = np.array([[1.0e-5, 4.0e-6], [4.0e-6, 2.5e-5]])
W = np.array([6e6, 4e6])


def test_eur_return_sign():
    # USD par EUR passe de 1,10 à 1,21 : l'euro s'apprécie, une position en dollars perd de la valeur.
    rates = pd.DataFrame({"USD": [1.10, 1.21]}, index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    assert eur_log_returns(rates)["USD"].iloc[0] == pytest.approx(-np.log(1.1))


def test_risk_from_pnl_quantile_and_es():
    pnl = -np.arange(1, 101, dtype=float)          # pertes 1 à 100
    risk = risk_from_pnl(pnl, 0.95)
    assert risk.var == pytest.approx(np.quantile(np.arange(1, 101), 0.95))
    assert risk.es >= risk.var


def test_monte_carlo_matches_gaussian_formula():
    rng = np.random.default_rng(0)
    mc = monte_carlo_risk(COV, W, 0.99, n_sims=400_000, rng=rng)
    exact = gaussian_risk(COV, W, 0.99)
    assert mc.var == pytest.approx(exact.var, rel=0.02)
    assert mc.es == pytest.approx(exact.es, rel=0.02)


def test_student_simulation_keeps_covariance_and_fattens_tails():
    rng = np.random.default_rng(1)
    draws = simulate_returns(COV, 400_000, rng, dist="student", nu=5)
    assert np.cov(draws.T) == pytest.approx(COV, rel=0.05)
    normal = monte_carlo_risk(COV, W, 0.999, 400_000, np.random.default_rng(2))
    student = monte_carlo_risk(COV, W, 0.999, 400_000, np.random.default_rng(2), "student", 5)
    assert student.var > normal.var


def test_student_requires_finite_variance():
    with pytest.raises(ValueError):
        simulate_returns(COV, 10, np.random.default_rng(0), dist="student", nu=2)


def test_fit_student_dof_recovers_parameter():
    x = np.random.default_rng(3).standard_t(5, size=50_000)
    assert fit_student_dof(x) == pytest.approx(5, abs=0.6)


def test_historical_uses_full_revaluation():
    # Log-rendement ln(0,9) : la position perd exactement 10 % (et non |ln 0,9| ≈ 10,5 %).
    window = np.log1p(np.full((3, 2), [-0.10, 0.0]))
    risk = historical_risk(window, np.array([1e6, 0.0]), 0.99)
    assert risk.var == pytest.approx(1e5)


def test_ewma_has_no_look_ahead():
    r = np.random.default_rng(4).normal(0, 0.01, size=(400, 2))
    base = ewma_covariances(r)
    shocked = r.copy()
    shocked[300] = 0.5
    after = ewma_covariances(shocked)
    assert np.allclose(base[300], after[300])        # le choc du jour 300 n'est pas encore connu le jour 300
    assert not np.allclose(base[301], after[301])
