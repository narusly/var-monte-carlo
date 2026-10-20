"""Taux de change de référence de la BCE : téléchargement, cache et rendements en euros."""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "ecb_fx_rates.csv"
CURRENCIES = ("USD", "GBP", "JPY", "CHF")
ECB_URL = ("https://data-api.ecb.europa.eu/service/data/EXR/D.{codes}.EUR.SP00.A"
           "?format=csvdata&startPeriod={start}")


def download_ecb_rates(currencies: tuple[str, ...] = CURRENCIES, start: str = "2010-01-01") -> pd.DataFrame:
    """Taux quotidiens « unités de devise pour 1 EUR », une colonne par devise, jours complets uniquement."""
    url = ECB_URL.format(codes="+".join(currencies), start=start)
    request = urllib.request.Request(url, headers={"User-Agent": "varmc"})
    with urllib.request.urlopen(request, timeout=120) as response:
        raw = pd.read_csv(io.BytesIO(response.read()), usecols=["CURRENCY", "TIME_PERIOD", "OBS_VALUE"])
    rates = raw.pivot(index="TIME_PERIOD", columns="CURRENCY", values="OBS_VALUE")
    rates.index = pd.to_datetime(rates.index)
    rates.index.name = "date"
    return rates[list(currencies)].dropna().sort_index()


def load_rates(refresh: bool = False) -> pd.DataFrame:
    """Lit le fichier versionné `data/ecb_fx_rates.csv` ; le (re)télécharge s'il manque ou si `refresh`."""
    if refresh or not DATA_FILE.exists():
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        download_ecb_rates().to_csv(DATA_FILE)
    return pd.read_csv(DATA_FILE, index_col="date", parse_dates=True)


def eur_log_returns(rates: pd.DataFrame) -> pd.DataFrame:
    """Log-rendements quotidiens, vus d'un investisseur en euros, d'une position en devise étrangère.

    La BCE cote S = unités de devise pour 1 EUR : une unité de devise vaut 1/S euros,
    donc son log-rendement en euros est -Δlog S (la devise s'apprécie quand S baisse).
    """
    return -np.log(rates).diff().dropna()
