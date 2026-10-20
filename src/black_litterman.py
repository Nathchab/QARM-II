"""
Black-Litterman (1992), as in He & Litterman (1999).

1. Prior: the returns that make the market portfolio optimal (reverse optimisation).
2. Views: the client says "I expect asset X to return q% per year", with a confidence.
3. Posterior: a mix of the prior and the views, weighted by their uncertainty.
"""
import numpy as np
import pandas as pd

# Approximate weight of each product in the global market portfolio, in %
# (rounded from Doeswijk, Lam & Swinkels 2014, "The Global Multi-Asset Market Portfolio").
# We only use the products chosen by the client and rescale them to 100%.
MARKET_WEIGHTS = {"SPY": 25, "QQQ": 5, "IWM": 2, "EFA": 10, "VGK": 4, "EWL": 1, "EWJ": 2, "EEM": 5,
                  "VNQ": 3, "AGG": 18, "SHY": 3, "TLT": 4, "TIP": 2, "LQD": 5, "HYG": 1, "EMB": 1,
                  "GLD": 3, "SLV": 0.5, "DBC": 2, "BTC-USD": 1, "ETH-USD": 0.5}
DELTA = 2.5  # risk aversion of the average investor (He & Litterman 1999)
TAU = 0.05   # uncertainty of the prior (small: the prior is quite reliable)

# QH6 of the questionnaire: confidence in a view -> c in Omega = tau * sigma^2 * (1 - c) / c
#   Low: 4x the prior uncertainty, Medium: same as the prior, High: 1/4, Very high: about 1/19
CONFIDENCE_LEVELS = {"Low": 0.2, "Medium": 0.5, "High": 0.8, "Very high / research-backed": 0.95}


def market_weights(tickers):
    w = pd.Series({t: MARKET_WEIGHTS[t] for t in tickers})
    return w / w.sum()


def prior_returns(cov, rf):
    """Equilibrium returns: pi = rf + delta * Cov * w_market."""
    return rf + DELTA * cov @ market_weights(cov.index)


def posterior_returns(cov, rf, views):
    """
    views: {ticker: (expected annual return, confidence between 0 and 1)}
    Returns the Black-Litterman expected returns.
    """
    pi = prior_returns(cov, rf)
    if not views:
        return pi  # no view: we keep the market equilibrium

    tickers = list(cov.index)
    P = np.zeros((len(views), len(tickers)))  # which asset each view is about
    Q = np.zeros(len(views))                  # the expected returns of the views
    omega = np.zeros(len(views))              # the uncertainty of each view
    for k, (ticker, (q, confidence)) in enumerate(views.items()):
        i = tickers.index(ticker)
        P[k, i] = 1
        Q[k] = q
        # confidence 50% -> same uncertainty as the prior; 95% -> almost certain
        omega[k] = TAU * cov.iloc[i, i] * (1 - confidence) / confidence

    tau_cov_inv = np.linalg.inv(TAU * cov.values)
    omega_inv = np.diag(1 / omega)
    A = tau_cov_inv + P.T @ omega_inv @ P
    b = tau_cov_inv @ pi.values + P.T @ omega_inv @ Q
    return pd.Series(np.linalg.solve(A, b), index=tickers)
