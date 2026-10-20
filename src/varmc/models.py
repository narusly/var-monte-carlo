"""Estimateurs de Value at Risk (VaR) et d'Expected Shortfall (ES) d'un portefeuille de devises.

Convention : VaR et ES sont des pertes, exprimées en euros et positives.
VaR_α = quantile α de la perte ; ES_α = perte moyenne au-delà de la VaR_α.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats


@dataclass(frozen=True)
class Risk:
    var: float
    es: float


def portfolio_pnl(returns: np.ndarray, exposures: np.ndarray) -> np.ndarray:
    """P&L en euros par réévaluation complète : chaque position vaut exposition × exp(r)."""
    return np.expm1(np.asarray(returns)) @ np.asarray(exposures)


def risk_from_pnl(pnl: np.ndarray, level: float) -> Risk:
    """VaR et ES empiriques d'un échantillon de P&L (historique ou simulé)."""
    losses = -np.asarray(pnl)
    var = float(np.quantile(losses, level))
    return Risk(var=var, es=float(losses[losses >= var].mean()))


def historical_risk(returns_window: np.ndarray, exposures: np.ndarray, level: float) -> Risk:
    """Simulation historique : on rejoue les rendements de la fenêtre sur les positions actuelles."""
    return risk_from_pnl(portfolio_pnl(returns_window, exposures), level)


def gaussian_risk(cov: np.ndarray, exposures: np.ndarray, level: float) -> Risk:
    """Méthode variance-covariance (delta-normale), moyenne supposée nulle à horizon 1 jour."""
    sigma = float(np.sqrt(exposures @ cov @ exposures))
    z = stats.norm.ppf(level)
    return Risk(var=z * sigma, es=sigma * stats.norm.pdf(z) / (1 - level))


def ewma_covariances(returns: np.ndarray, lam: float = 0.94, init_window: int = 250) -> np.ndarray:
    """Matrices de covariance RiskMetrics (λ = 0,94).

    `out[t]` n'utilise que les rendements jusqu'à t-1 : c'est la prévision disponible au début du jour t.
    L'initialisation sur les `init_window` premiers jours n'est valable qu'à partir de t ≥ init_window.
    """
    r = np.asarray(returns)
    out = np.empty((len(r), r.shape[1], r.shape[1]))
    s = np.cov(r[:init_window].T)
    for t in range(len(r)):
        out[t] = s
        s = lam * s + (1 - lam) * np.outer(r[t], r[t])
    return out


def simulate_returns(cov: np.ndarray, n_sims: int, rng: np.random.Generator,
                     dist: str = "normal", nu: float | None = None) -> np.ndarray:
    """Tire `n_sims` vecteurs de rendements de covariance `cov` (normale ou Student multivariée)."""
    chol = np.linalg.cholesky(cov)
    draws = rng.standard_normal((n_sims, cov.shape[0])) @ chol.T
    if dist == "student":
        if nu is None or nu <= 2:
            raise ValueError("une loi de Student à variance finie demande nu > 2")
        # Z·√((ν-2)/ν) / √(W/ν) avec W ~ χ²(ν) : Student multivariée de covariance exactement `cov`.
        chi2 = rng.chisquare(nu, size=(n_sims, 1))
        draws = draws * np.sqrt((nu - 2) / nu) / np.sqrt(chi2 / nu)
    elif dist != "normal":
        raise ValueError(f"loi inconnue : {dist}")
    return draws


def monte_carlo_risk(cov: np.ndarray, exposures: np.ndarray, level: float, n_sims: int = 50_000,
                     rng: np.random.Generator | None = None, dist: str = "normal",
                     nu: float | None = None) -> Risk:
    """VaR et ES par simulation de Monte Carlo et réévaluation complète du portefeuille."""
    rng = rng if rng is not None else np.random.default_rng()
    sims = simulate_returns(cov, n_sims, rng, dist, nu)
    return risk_from_pnl(portfolio_pnl(sims, exposures), level)


def fit_student_dof(x: np.ndarray) -> float:
    """Degrés de liberté ν d'une loi de Student réduite (variance 1), par maximum de vraisemblance."""
    x = np.asarray(x, dtype=float)
    x = (x - x.mean()) / x.std()

    def negative_log_likelihood(nu: float) -> float:
        scale = np.sqrt((nu - 2) / nu)
        return -stats.t.logpdf(x / scale, nu).sum() + len(x) * np.log(scale)

    result = optimize.minimize_scalar(negative_log_likelihood, bounds=(2.05, 100.0), method="bounded")
    return float(result.x)


def var_confidence_interval(n_sims: int, level: float, sorted_losses: np.ndarray,
                            confidence: float = 0.95) -> tuple[float, float]:
    """Intervalle de confiance de la VaR Monte Carlo par statistiques d'ordre (loi binomiale), sans hypothèse de loi."""
    lo_rank = int(stats.binom.ppf((1 - confidence) / 2, n_sims, level))
    hi_rank = int(stats.binom.ppf(1 - (1 - confidence) / 2, n_sims, level))
    return float(sorted_losses[max(lo_rank - 1, 0)]), float(sorted_losses[min(hi_rank, n_sims - 1)])
