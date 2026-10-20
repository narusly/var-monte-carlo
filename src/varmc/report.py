"""Analyse complète : `python -m varmc.report` écrit `reports/resultats.md` et les figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy import stats

from . import style
from .backtest import MODELS, evaluate, rolling_var
from .data import ROOT, eur_log_returns, load_rates
from .models import (ewma_covariances, fit_student_dof, gaussian_risk, historical_risk,
                     monte_carlo_risk, portfolio_pnl, simulate_returns, var_confidence_interval)

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
# Portefeuille d'un investisseur en euros : 10 M€ répartis sur quatre devises.
EXPOSURES = pd.Series({"USD": 4_000_000, "GBP": 2_500_000, "JPY": 2_000_000, "CHF": 1_500_000})
WINDOW, SEED = 500, 42

keur = FuncFormatter(lambda v, _: f"{v / 1000:,.0f} k€".replace(",", " "))
# Événements de marché vérifiés dans les données, annotés s'ils correspondent à un extrême du P&L.
EVENTS = {pd.Timestamp("2015-01-15"): "15/01/2015 : la BNS abandonne\nle taux plancher EUR/CHF"}


def fr(x: float, decimals: int = 1) -> str:
    """Nombre au format français : espace des milliers, virgule décimale."""
    return f"{x:,.{decimals}f}".replace(",", " ").replace(".", ",")


def fr_pct(x: float, decimals: int = 2) -> str:
    return f"{fr(100 * x, decimals)} %"


def label_line_ends(ax, points: list[tuple[str, pd.Timestamp, float]], min_gap: float) -> None:
    """Étiquettes directes en bout de courbe, écartées verticalement si elles se chevauchent."""
    placed: list[float] = []
    for text, x, y in sorted(points, key=lambda p: p[2]):
        y_text = max(y, placed[-1] + min_gap) if placed else y
        placed.append(y_text)
        x_num = mdates.date2num(x)
        ax.annotate(text, (x_num, y), xytext=(x_num + 40, y_text), textcoords="data", va="center",
                    fontsize=8.5, color=style.INK_2, annotation_clip=False)


def current_risk(returns: pd.DataFrame, w: np.ndarray) -> pd.DataFrame:
    """VaR et ES à la dernière date, pour chaque modèle, à 99 % et 97,5 %."""
    r = returns.to_numpy()
    past = r[-WINDOW:]
    ewma_next = ewma_covariances(np.vstack([r, np.zeros(r.shape[1])]))[-1]   # prévision pour le jour suivant
    pnl_hist = portfolio_pnl(r, w)
    sigma = np.sqrt(np.einsum("i,tij,j->t", w, ewma_covariances(r), w))
    nu = fit_student_dof((pnl_hist / sigma)[-WINDOW:])
    rng = np.random.default_rng(SEED)
    rows = []
    for level in (0.99, 0.975):
        estimates = {
            "hs": historical_risk(past, w, level),
            "gaussian": gaussian_risk(np.cov(past.T), w, level),
            "mc_normal_ewma": monte_carlo_risk(ewma_next, w, level, 200_000, rng),
            "mc_student_ewma": monte_carlo_risk(ewma_next, w, level, 200_000, rng, "student", nu),
        }
        for key, risk in estimates.items():
            rows.append({"niveau": fr_pct(level, 1).replace(",0 %", " %"), "modèle": MODELS[key],
                         "VaR (€)": risk.var, "ES (€)": risk.es})
    table = pd.DataFrame(rows)
    table.attrs["nu"] = nu
    table.attrs["ewma_next"] = ewma_next
    return table


def fig_backtest(bt: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(bt.index, bt["pnl"], color=style.MUTED, linewidth=0.4, alpha=0.55, label="P&L quotidien réalisé")
    ends = []
    for key, color in (("hs", style.SERIES[0]), ("mc_student_ewma", style.SERIES[1])):
        ax.plot(bt.index, -bt[key], color=color, linewidth=1.3, label=f"−VaR 99 % · {MODELS[key]}")
        ends.append((MODELS[key].split(" (")[0], bt.index[-1], -bt[key].iloc[-1]))
    label_line_ends(ax, ends, min_gap=22_000)
    for day, text in EVENTS.items():
        if day in bt.index and bt["pnl"].idxmax() == day:
            ax.annotate(text, (day, bt.loc[day, "pnl"]), xytext=(14, -4), textcoords="offset points",
                        va="top", fontsize=8.5, color=style.INK_2)
    hits = bt["pnl"] < -bt["mc_student_ewma"]
    ax.scatter(bt.index[hits], bt["pnl"][hits], s=14, color=style.CRITICAL, zorder=3,
               edgecolors=style.SURFACE, linewidths=0.6, label="Exception (Monte Carlo Student + EWMA)")
    ax.axhline(0, color=style.BASELINE, linewidth=0.8)
    ax.yaxis.set_major_formatter(keur)
    ax.set_title("Backtesting de la VaR 99 % à 1 jour, portefeuille de 10 M€ en devises")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncols=2)
    ax.margins(x=0.01)
    fig.savefig(path)
    plt.close(fig)


def fig_distribution(cov: np.ndarray, w: np.ndarray, nu: float, path: Path) -> None:
    rng = np.random.default_rng(SEED)
    pnl_t = portfolio_pnl(simulate_returns(cov, 200_000, rng, "student", nu), w)
    pnl_n = portfolio_pnl(simulate_returns(cov, 200_000, rng, "normal"), w)
    var_t, var_n = np.quantile(-pnl_t, 0.99), np.quantile(-pnl_n, 0.99)
    es_t = (-pnl_t)[-pnl_t >= var_t].mean()
    lo = np.quantile(pnl_t, 0.0005)
    bins = np.linspace(lo, -lo, 140)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.hist(pnl_t, bins=bins, density=True, color=style.SERIES[0], alpha=0.85, rwidth=0.9,
            label=f"Monte Carlo Student (ν = {fr(nu)})")
    ax.hist(pnl_n, bins=bins, density=True, histtype="step", color=style.SERIES[1], linewidth=1.4,
            label="Monte Carlo normal")
    for x, text, dy in ((-var_n, f"VaR 99 % normale\n{var_n / 1000:,.0f} k€", 0.62),
                        (-var_t, f"VaR 99 % Student\n{var_t / 1000:,.0f} k€", 0.40),
                        (-es_t, f"ES 99 % Student\n{es_t / 1000:,.0f} k€", 0.18)):
        ax.axvline(x, color=style.INK_2, linewidth=0.9)
        ax.annotate(text.replace(",", " "), (x, ax.get_ylim()[1] * dy), xytext=(-6, 0), textcoords="offset points",
                    ha="right", fontsize=8.5, color=style.INK_2)
    ax.xaxis.set_major_formatter(keur)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set_title("Distribution du P&L à 1 jour simulée à la dernière date")
    ax.set_xlabel("P&L du portefeuille")
    ax.legend(loc="upper right")
    fig.savefig(path)
    plt.close(fig)


def fig_convergence(cov: np.ndarray, w: np.ndarray, path: Path) -> None:
    """Estimation Monte Carlo de la VaR 99 % (loi normale) selon le nombre de tirages, avec IC à 95 %."""
    rng = np.random.default_rng(SEED)
    sizes = np.unique(np.logspace(2.5, 6, 22).astype(int))
    est, lo, hi = [], [], []
    for n in sizes:
        losses = np.sort(-portfolio_pnl(simulate_returns(cov, n, rng), w))
        est.append(np.quantile(losses, 0.99))
        a, b = var_confidence_interval(n, 0.99, losses)
        lo.append(a)
        hi.append(b)
    exact = gaussian_risk(cov, w, 0.99).var
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.fill_between(sizes, lo, hi, color=style.SERIES[0], alpha=0.18, linewidth=0, label="Intervalle de confiance à 95 %")
    ax.plot(sizes, est, color=style.SERIES[0], marker="o", markersize=4, label="VaR 99 % estimée par Monte Carlo")
    ax.axhline(exact, color=style.INK_2, linewidth=0.9)
    ax.annotate(f"Valeur analytique (loi normale) : {fr(exact / 1000)} k€", (sizes[-1], exact),
                xytext=(0, 9), textcoords="offset points", ha="right", fontsize=8.5, color=style.INK_2)
    ax.set_xscale("log")
    ax.yaxis.set_major_formatter(keur)
    ax.set_xlabel("Nombre de tirages")
    ax.set_title("Convergence de l'estimateur Monte Carlo de la VaR 99 %")
    ax.legend(loc="upper right")
    fig.savefig(path)
    plt.close(fig)


def fig_qq(pnl: pd.Series, sigma: np.ndarray, path: Path) -> None:
    standardized = pnl / sigma
    order = np.argsort(standardized.to_numpy())
    z = standardized.to_numpy()[order]
    z = (z - z.mean()) / z.std()
    theo = stats.norm.ppf((np.arange(1, len(z) + 1) - 0.5) / len(z))
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.plot(theo, theo, color=style.BASELINE, linewidth=1)
    ax.scatter(theo, z, s=8, color=style.SERIES[0], edgecolors="none", alpha=0.7)
    ax.annotate("Queues plus épaisses\nque la loi normale", (theo[3], z[3]), xytext=(18, -6), textcoords="offset points",
                fontsize=8.5, color=style.INK_2)
    top_day = standardized.index[order[-1]]
    if top_day in EVENTS:
        ax.annotate(EVENTS[top_day].replace("\n", " ").replace(" : ", "\n", 1), (theo[-1], z[-1]), xytext=(-8, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=8.5, color=style.INK_2)
    ax.set_xlabel("Quantiles théoriques N(0, 1)")
    ax.set_ylabel("Quantiles observés")
    ax.set_title("Q-Q plot des rendements réduits par l'EWMA")
    fig.savefig(path)
    plt.close(fig)


def fmt_eur(x: float) -> str:
    return f"{x:,.0f} €".replace(",", " ")


def main() -> None:
    style.apply()
    FIGURES.mkdir(parents=True, exist_ok=True)
    rates = load_rates()
    returns = eur_log_returns(rates)[EXPOSURES.index]
    w = EXPOSURES.to_numpy(dtype=float)

    risk = current_risk(returns, w)
    bt = rolling_var(returns, w, level=0.99, window=WINDOW, seed=SEED)
    scores = evaluate(bt, level=0.99)

    r = returns.to_numpy()
    pnl = portfolio_pnl(r, w)
    sigma = np.sqrt(np.einsum("i,tij,j->t", w, ewma_covariances(r), w))
    fig_backtest(bt, FIGURES / "backtest.png")
    fig_distribution(risk.attrs["ewma_next"], w, risk.attrs["nu"], FIGURES / "distribution.png")
    fig_convergence(risk.attrs["ewma_next"], w, FIGURES / "convergence.png")
    fig_qq(pd.Series(pnl, index=returns.index)[250:], sigma[250:], FIGURES / "qqplot.png")

    risk_md = risk.assign(**{"VaR (€)": risk["VaR (€)"].map(fmt_eur), "ES (€)": risk["ES (€)"].map(fmt_eur)})
    scores_md = scores.assign(**{
        "attendues": scores["attendues"].map(lambda x: fr(x, 0)),
        "taux": scores["taux"].map(fr_pct),
        "p Kupiec": scores["p Kupiec"].map(lambda x: fr(x, 3)),
        "p Christoffersen": scores["p Christoffersen"].map(lambda x: fr(x, 3)),
    })
    kurt = stats.kurtosis(pnl[250:] / sigma[250:], fisher=True)
    lines = [
        "# Résultats",
        "",
        f"Données BCE du {returns.index[0]:%d/%m/%Y} au {returns.index[-1]:%d/%m/%Y} "
        f"({fr(len(returns), 0)} jours ouvrés). Portefeuille : "
        + ", ".join(f"{c} {fr(v / 1e6)} M€" for c, v in EXPOSURES.items()) + ".",
        "",
        f"Kurtosis en excès des rendements réduits par l'EWMA : {fr(kurt, 2)} (0 pour une loi normale). "
        f"Degrés de liberté de la Student estimés à la dernière date : ν = {fr(risk.attrs['nu'])}.",
        "",
        "## Risque à la dernière date (horizon 1 jour)",
        "",
        risk_md.to_markdown(index=False),
        "",
        f"## Backtesting VaR 99 % ({bt.index[0]:%d/%m/%Y} – {bt.index[-1]:%d/%m/%Y})",
        "",
        scores_md.to_markdown(index=False),
        "",
    ]
    (REPORTS / "resultats.md").write_text("\n".join(lines), encoding="utf-8")
    bt.to_csv(REPORTS / "backtest.csv", float_format="%.2f")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
