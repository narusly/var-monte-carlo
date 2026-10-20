"""Backtesting de la VaR : VaR glissante, tests de Kupiec et de Christoffersen, feux de Bâle."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .models import (ewma_covariances, fit_student_dof, gaussian_risk, historical_risk,
                     monte_carlo_risk, portfolio_pnl)

MODELS = {
    "hs": "Simulation historique (500 j)",
    "gaussian": "Variance-covariance (500 j)",
    "mc_normal_ewma": "Monte Carlo normal + EWMA",
    "mc_student_ewma": "Monte Carlo Student + EWMA",
}


def _xlogy(x: float, y: float) -> float:
    return 0.0 if x == 0 else x * np.log(y)


def kupiec_pof(n_exceptions: int, n_obs: int, tail: float) -> tuple[float, float]:
    """Test de Kupiec (proportion of failures) : la fréquence des exceptions vaut-elle `tail` ?

    Renvoie (statistique LR, p-value) ; LR suit un χ²(1) sous H0.
    """
    x, n = n_exceptions, n_obs
    log_h0 = _xlogy(n - x, 1 - tail) + _xlogy(x, tail)
    p_hat = x / n
    log_h1 = _xlogy(n - x, 1 - p_hat) + _xlogy(x, p_hat)
    lr = -2 * (log_h0 - log_h1)
    return lr, float(stats.chi2.sf(lr, 1))


def christoffersen_independence(exceptions: np.ndarray) -> tuple[float, float]:
    """Test d'indépendance de Christoffersen : une exception rend-elle la suivante plus probable ?

    Compare une chaîne de Markov d'ordre 1 à une loi de Bernoulli. Renvoie (LR, p-value), χ²(1) sous H0.
    """
    e = np.asarray(exceptions, dtype=int)
    prev, curr = e[:-1], e[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    log_h0 = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    log_h1 = _xlogy(n00, 1 - pi01) + _xlogy(n01, pi01) + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11)
    lr = -2 * (log_h0 - log_h1)
    return lr, float(stats.chi2.sf(lr, 1))


def basel_zone(n_exceptions_250: int) -> str:
    """Feux de Bâle pour une VaR à 99 % sur 250 jours : vert ≤ 4, orange 5 à 9, rouge ≥ 10."""
    if n_exceptions_250 <= 4:
        return "vert"
    return "orange" if n_exceptions_250 <= 9 else "rouge"


def rolling_var(returns: pd.DataFrame, exposures: np.ndarray, level: float = 0.99, window: int = 500,
                n_sims: int = 20_000, seed: int = 42) -> pd.DataFrame:
    """VaR de chaque modèle calculée chaque jour avec les seules données antérieures, et P&L réalisé."""
    r = returns.to_numpy()
    rng = np.random.default_rng(seed)
    ewma = ewma_covariances(r)
    realized = portfolio_pnl(r, exposures)
    # Rendements du portefeuille réduits par la volatilité EWMA de la veille : ils servent à estimer ν.
    sigma_ewma = np.sqrt(np.einsum("i,tij,j->t", exposures, ewma, exposures))
    standardized = realized / sigma_ewma

    rows = []
    for t in range(window, len(r)):
        past = r[t - window:t]
        nu = fit_student_dof(standardized[t - window:t])
        rows.append({
            "pnl": realized[t],
            "hs": historical_risk(past, exposures, level).var,
            "gaussian": gaussian_risk(np.cov(past.T), exposures, level).var,
            "mc_normal_ewma": monte_carlo_risk(ewma[t], exposures, level, n_sims, rng).var,
            "mc_student_ewma": monte_carlo_risk(ewma[t], exposures, level, n_sims, rng, "student", nu).var,
            "nu": nu,
        })
    return pd.DataFrame(rows, index=returns.index[window:])


def evaluate(backtest: pd.DataFrame, level: float = 0.99) -> pd.DataFrame:
    """Exceptions, tests de Kupiec et de Christoffersen, et zone de Bâle sur les 250 derniers jours."""
    tail = 1 - level
    rows = []
    for key, label in MODELS.items():
        exceptions = (backtest["pnl"] < -backtest[key]).to_numpy()
        n, x = len(exceptions), int(exceptions.sum())
        _, p_kupiec = kupiec_pof(x, n, tail)
        _, p_indep = christoffersen_independence(exceptions)
        rows.append({
            "modèle": label,
            "jours": n,
            "exceptions": x,
            "attendues": round(n * tail, 1),
            "taux": x / n,
            "p Kupiec": p_kupiec,
            "p Christoffersen": p_indep,
            "exceptions 250 j": int(exceptions[-250:].sum()),
            "zone Bâle": basel_zone(int(exceptions[-250:].sum())),
        })
    return pd.DataFrame(rows)
