"""
The four portfolios we offer to the client, and the benchmarks.
Used by the app pages and by the backtest.
"""
import numpy as np
import pandas as pd

from src.black_litterman import market_weights, posterior_returns, restrict_views
from src.data import ASSETS
from src.optimizer import gamma_for_max_volatility, mean_variance, portfolio_stats, resampled, risk_parity

WINDOW = 60  # we estimate returns and risk on the last 60 months (5 years)
METHODS = ["Black-Litterman", "Mean-variance", "Resampled MV", "Risk parity (ERC)"]
RECOMMENDED_METHOD = "Black-Litterman"  # without views = market equilibrium returns, the most stable inputs
# "Loss in a bad year" = expected return - 1.645 x volatility: with a normal distribution, a year is worse than
# that with a probability of 5%. The client's answer to QE2 is the largest loss he can tolerate.
BAD_YEAR = 1.645

BENCHMARKS = (["Same equity share as your portfolio (SPY + AGG)", "60/40 (60% SPY, 40% AGG)",
               "Equal weights (1/N) of your products",
               "Market portfolio of your products"]
              + [f"100% {t} ({name})" for t, (name, _) in ASSETS.items()])


def estimate(monthly_returns):
    """Annual expected returns and covariance matrix from the last WINDOW months."""
    past = monthly_returns.dropna().iloc[-WINDOW:]
    return past.mean() * 12, past.cov() * 12


def build_portfolios(mu, cov, rf, gamma, c, views, n_sim=100):
    """The four methods, all with the same client constraints.
    Returns (weights of the methods that worked, error message of the methods that failed):
    one failing method does not hide the others."""
    views = restrict_views(views, mu.index)  # ignore views on products not chosen
    methods = {
        "Mean-variance": lambda: mean_variance(mu, cov, gamma, c),
        "Resampled MV": lambda: resampled(mu, cov, gamma, c, n_sim),
        "Black-Litterman": lambda: mean_variance(posterior_returns(cov, rf, views), cov, gamma, c),
        "Risk parity (ERC)": lambda: risk_parity(cov, c),
    }
    portfolios, errors = {}, {}
    for name, compute in methods.items():
        try:
            portfolios[name] = compute()
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            errors[name] = str(error)
    return portfolios, errors


def bad_year_loss(weights, forecast, cov):
    """Loss in a bad year (5% chance of doing worse), e.g. -0.08 = -8%."""
    ret, vol = portfolio_stats(weights, forecast, cov)
    return ret - BAD_YEAR * vol


def gamma_for_loss_limit(forecast, cov, gamma, c, max_loss):
    """Risk aversion actually used: the client's gamma, raised if needed so that the recommended portfolio's
    loss in a bad year stays within his answer to QE2 (max_loss, e.g. -0.10; None = more than 30%).
    We ignore the expected return (prudent), so the volatility must be at most |max_loss| / 1.645."""
    if max_loss is None:
        return gamma
    return gamma_for_max_volatility(forecast, cov, gamma, c, -max_loss / BAD_YEAR)


def method_name(method, views):
    """Name shown to the client: Black-Litterman without views is simply the market equilibrium."""
    if method == "Black-Litterman":
        return "Black-Litterman with your views" if views else "Market equilibrium (Black-Litterman)"
    return method


def dominated_by(portfolios, forecast, cov, method):
    """Methods whose portfolio has a higher expected return AND a lower risk than `method`, with our forecast."""
    ret, vol = portfolio_stats(portfolios[method], forecast, cov)
    better = []
    for other, w in portfolios.items():
        r, v = portfolio_stats(w, forecast, cov)
        if r >= ret - 1e-9 and v <= vol + 1e-9 and (r > ret + 1e-6 or v < vol - 1e-6):
            better.append(other)
    return better


def recommended_method(portfolios, forecast, cov):
    """Black-Litterman, unless another method dominates it: then the dominating method with the closest risk.
    (With our forecast, Black-Litterman is the optimal portfolio, so in practice it is never dominated.)"""
    if RECOMMENDED_METHOD not in portfolios:
        return next(iter(portfolios))
    better = dominated_by(portfolios, forecast, cov, RECOMMENDED_METHOD)
    if not better:
        return RECOMMENDED_METHOD
    vol = portfolio_stats(portfolios[RECOMMENDED_METHOD], forecast, cov)[1]
    return min(better, key=lambda m: abs(portfolio_stats(portfolios[m], forecast, cov)[1] - vol))


def benchmark_weights(name, tickers, equity_share):
    """Weights of the benchmark chosen by the client. equity_share: equities in the recommended portfolio."""
    if name.startswith("Same equity share"):  # the simplest portfolio with the same stock/bond split as ours
        return pd.Series({"SPY": equity_share, "AGG": 1 - equity_share})
    if name.startswith("60/40"):
        return pd.Series({"SPY": 0.6, "AGG": 0.4})
    if name.startswith("Equal"):
        return pd.Series(1 / len(tickers), index=tickers)
    if name.startswith("Market"):
        return market_weights(tickers)
    return pd.Series({name.split()[1]: 1.0})  # "100% SPY (...)" -> SPY
