"""
Black-Litterman (lecture LN4; Black & Litterman 1992, He & Litterman 1999).

1. Prior: the returns that make the market portfolio optimal (reverse optimisation).
2. Views: "asset i will return q% per year" (absolute) or "asset i will beat asset j by q% per year"
   (relative), each with a confidence.
3. Posterior: a mix of the prior and the views, weighted by their uncertainty.

Black-Litterman is only used for experienced investors who enter views. Every other client gets the prior
alone (the "market equilibrium" method), because it is much more stable than 60-month historical averages.

A view is a tuple (asset, other, q, confidence):
    absolute: ("GLD", None, 0.06, 0.5)   -> P row: +1 on GLD,             Q = 6%
    relative: ("EEM", "SPY", 0.03, 0.8)  -> P row: +1 on EEM, -1 on SPY,  Q = 3%
"""
import numpy as np
import pandas as pd

# Weight of each product in the "market portfolio", in %. These are OUR OWN round numbers: stocks and
# bonds dominate, as in Doeswijk, Lam & Swinkels (2014), but we did not copy their figures (they have
# about 36% in equities, and no gold or crypto). ETFs that overlap (QQQ is inside SPY, everything is inside
# ACWI) are a simplification.
# We only use the products chosen by the client and rescale them to 100%.
MARKET_WEIGHTS = {"SPY": 25, "QQQ": 5, "IWM": 2, "ACWI": 40, "EFA": 10, "VGK": 4, "EZU": 3, "EWU": 1.5, "EWG": 1,
                  "CSSMI.SW": 1, "CSSMIM.SW": 0.3, "EWL": 1, "EWJ": 2, "EEM": 5, "VNQ": 3,
                  "AGG": 18, "SHY": 3, "TLT": 4, "TIP": 2, "LQD": 5, "HYG": 1, "EMB": 1, "BNDX": 10, "BWX": 3,
                  "CSBGC3.SW": 1, "CSBGC7.SW": 1.5, "CSBGC0.SW": 1, "CHCORP.SW": 1,
                  "GLD": 3, "SLV": 0.5, "DBC": 2, "BTC-USD": 1, "ETH-USD": 0.5}
DELTA = 2.5  # risk aversion of the average investor (He & Litterman 1999)
TAU = 0.05   # uncertainty of the prior: Gamma = tau * Sigma in LN4 (small: the prior is quite reliable)

# QH6 of the questionnaire: confidence in a view -> c in Omega_kk = tau * (P Sigma P')_kk * (1 - c) / c
#   Low: 4x the prior uncertainty, Medium: same as the prior, High: 1/4, Very high: about 1/19
CONFIDENCE_LEVELS = {"Low": 0.2, "Medium": 0.5, "High": 0.8, "Very high / research-backed": 0.95}
MAX_VIEWS = 5
EXTREME_VIEW = 2.0  # warn when a view is more than 2 standard deviations away from the equilibrium


def market_weights(tickers):
    w = pd.Series({t: MARKET_WEIGHTS[t] for t in tickers})
    return w / w.sum()


def prior_returns(cov, rf):
    """Equilibrium returns: pi = rf + delta * Cov * w_market."""
    return rf + DELTA * cov @ market_weights(cov.index)


def restrict_views(views, tickers):
    """Keep only the views on products that are in the portfolio."""
    return tuple(tuple(v) for v in views if v[0] in tickers and (v[1] is None or v[1] in tickers))


def view_problems(views):
    """Problems that make a set of views invalid (empty list = valid)."""
    problems = []
    if len(views) > MAX_VIEWS:
        problems.append(f"At most {MAX_VIEWS} views.")
    seen = set()
    for asset, other, q, confidence in views:
        if other == asset:
            problems.append(f"A relative view needs two different products ({asset}).")
        key = asset if other is None else frozenset((asset, other))
        if key in seen:
            problems.append(f"Two views on the same {'product' if other is None else 'pair'}: "
                            f"{asset if other is None else asset + ' / ' + other}. Keep only one.")
        seen.add(key)
        if not 0 < confidence < 1:
            problems.append("Confidence must be strictly between 0% and 100%.")
    return problems


def view_row(view, tickers):
    """Row of P for one view: +1 on the asset, and -1 on the other asset of a relative view."""
    asset, other, q, confidence = view
    row = np.zeros(len(tickers))
    row[tickers.index(asset)] = 1.0
    if other is not None:
        row[tickers.index(other)] = -1.0
    return row


def view_check(cov, pi, view):
    """Equilibrium value of the view (P pi), its volatility sqrt(P Sigma P') and the gap in standard deviations."""
    row = view_row(view, list(cov.index))
    equilibrium = float(row @ pi.values)
    sd = float(np.sqrt(row @ cov.values @ row))
    gap = (view[2] - equilibrium) / sd if sd > 0 else 0.0
    return {"equilibrium": equilibrium, "volatility": sd, "gap": gap, "extreme": abs(gap) > EXTREME_VIEW}


def posterior_returns(cov, rf, views):
    """Black-Litterman expected returns: the equilibrium returns corrected by the client's (valid) views."""
    pi = prior_returns(cov, rf)
    if not views:
        return pi  # no view: we keep the market equilibrium

    tickers = list(cov.index)
    P = np.array([view_row(v, tickers) for v in views])  # which assets each view is about
    Q = np.array([v[2] for v in views])                   # the expected returns of the views
    confidence = np.array([v[3] for v in views])
    gamma_matrix = TAU * cov.values                       # uncertainty of the prior (LN4: Gamma = tau * Sigma)
    # Uncertainty of each view: confidence 50% -> same uncertainty as the prior; 95% -> almost certain
    omega = np.diag(np.diag(P @ gamma_matrix @ P.T) * (1 - confidence) / confidence)
    # LN4: mu_bar = pi + Gamma P' (P Gamma P' + Omega)^-1 (Q - P pi)
    disequilibrium = Q - P @ pi.values                    # gap between the views and the market
    correction = gamma_matrix @ P.T @ np.linalg.solve(P @ gamma_matrix @ P.T + omega, disequilibrium)
    return pi + correction
