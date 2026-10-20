"""
The four portfolios we offer to the client, and the benchmarks.
Used by the app pages and by the backtest.
"""
import numpy as np
import pandas as pd

from src.black_litterman import market_weights, posterior_returns
from src.data import ASSETS
from src.optimizer import mean_variance, resampled, risk_parity

WINDOW = 60  # we estimate returns and risk on the last 60 months (5 years)
METHODS = ["Mean-variance", "Resampled MV", "Black-Litterman", "Risk parity (ERC)"]

BENCHMARKS = (["Your profile: SPY up to your equity maximum, the rest AGG", "60/40 (60% SPY, 40% AGG)",
               "Equal weights (1/N) of your products",
               "Market portfolio of your products"]
              + [f"100% {t} ({name})" for t, (name, _) in ASSETS.items()])


def estimate(monthly_returns):
    """Annual expected returns and covariance matrix from the last WINDOW months."""
    past = monthly_returns.dropna().iloc[-WINDOW:]
    return past.mean() * 12, past.cov() * 12


def build_portfolios(mu, cov, rf, gamma, c, views, n_sim=100):
    """The weights of the four methods, all with the same client constraints."""
    views = {t: v for t, v in views.items() if t in mu.index}  # ignore views on products not chosen
    return {
        "Mean-variance": mean_variance(mu, cov, gamma, c),
        "Resampled MV": resampled(mu, cov, gamma, c, n_sim),
        "Black-Litterman": mean_variance(posterior_returns(cov, rf, views), cov, gamma, c),
        "Risk parity (ERC)": risk_parity(cov, c),
    }


def build_portfolios_safe(mu, cov, rf, gamma, c, views, n_sim=100):
    """Same as build_portfolios, but one failing method does not hide the others.
    Returns (weights of the methods that worked, error message of the methods that failed)."""
    views = {t: v for t, v in views.items() if t in mu.index}
    calls = {
        "Mean-variance": lambda: mean_variance(mu, cov, gamma, c),
        "Resampled MV": lambda: resampled(mu, cov, gamma, c, n_sim),
        "Black-Litterman": lambda: mean_variance(posterior_returns(cov, rf, views), cov, gamma, c),
        "Risk parity (ERC)": lambda: risk_parity(cov, c),
    }
    portfolios, errors = {}, {}
    for name, compute in calls.items():
        try:
            portfolios[name] = compute()
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            errors[name] = str(error)
    return portfolios, errors


def benchmark_weights(name, tickers, c):
    """Weights of the benchmark chosen by the client."""
    if name.startswith("Your profile"):  # a simple stock/bond mix with the same equity limit as the client
        return pd.Series({"SPY": c["max_equity"], "AGG": 1 - c["max_equity"]})
    if name.startswith("60/40"):
        return pd.Series({"SPY": 0.6, "AGG": 0.4})
    if name.startswith("Equal"):
        return pd.Series(1 / len(tickers), index=tickers)
    if name.startswith("Market"):
        return market_weights(tickers)
    return pd.Series({name.split()[1]: 1.0})  # "100% SPY (...)" -> SPY
