"""
The portfolios we offer to the client, and the benchmarks.
Used by the app pages and by the backtest.

Which method we recommend depends on the client (recommended_method):
  - clients who need the money within a year, or may withdraw more than half of it within 12 months ->
    Minimum variance: over such a short time the expected return hardly matters, only the risk does, and it
    needs no forecast of returns (no estimation error on mu);
  - experienced investors who entered their own views (QH4) -> Black-Litterman: it moves the equilibrium
    returns towards their views (LN4);
  - clients whose risk answers contradict each other (QE3 far from the other answers) -> Risk parity (ERC, LN5):
    we cannot measure their gamma reliably, and risk parity does not need gamma (nor expected returns).
    Only if it respects their loss limit (QE2), otherwise Market equilibrium;
  - everybody else -> "Market equilibrium": mean-variance (Markowitz) with the expected returns implied by the
    market (pi_i = rf + delta * cov(r_i, r_market), "reverse optimisation"), because the average of the last
    5 years is a very noisy forecast (LN3).
Mean-variance and resampled MV with past returns are shown for comparison only.
"""
import numpy as np
import pandas as pd

from src.black_litterman import MARKET_PORTFOLIO, posterior_returns, restrict_views
from src.data import ASSETS
from src.optimizer import (class_exposures, gamma_for, gamma_for_max_volatility, mean_variance, minimum_variance, portfolio_stats, resampled,
                           risk_parity)

WINDOW = 60  # we estimate returns and risk on the last 60 months (5 years)
METHODS = ["Equilibrium", "Minimum variance", "Black-Litterman", "Mean-variance", "Resampled MV", "Risk parity (ERC)"]
LABELS = {"Equilibrium": "Market equilibrium (mean-variance)", "Minimum variance": "Minimum variance",
          "Black-Litterman": "Black-Litterman (your views)",
          "Mean-variance": "Mean-variance (past returns)", "Resampled MV": "Resampled MV (past returns)",
          "Risk parity (ERC)": "Risk parity (ERC)"}
# "Loss in a bad year" = expected return - 1.645 x volatility: with a normal distribution, a year is worse than
# that with a probability of 5%. The client's answer to QE2 is the largest loss he can tolerate.
BAD_YEAR = 1.645

# Default benchmark: the simplest portfolio with the same share of equities as ours, that the client could buy
# himself: the S&P 500 and government bonds in his currency
BENCHMARK_BOND = {"USD": "AGG", "CHF": "CSBGC7.SW"}
BENCHMARKS = (["Same equity share as your portfolio (S&P 500 + bonds in your currency)", "60/40 (60% SPY, 40% AGG)",
               "Equal weights (1/N) of your products",
               "Market portfolio (48% equities, 45% bonds, 7% gold and commodities)"]
              + [f"100% {t} ({name})" for t, (name, _) in ASSETS.items()])


def inputs(past, tickers):
    """Inputs of the optimisers from a window of monthly returns (all annual): the average returns and the
    covariance matrix of the client's products, and their covariance with the market portfolio."""
    market = past[list(MARKET_PORTFOLIO)] @ pd.Series(MARKET_PORTFOLIO)
    return past[tickers].mean() * 12, past[tickers].cov() * 12, past[tickers].apply(market.cov) * 12


def estimate(monthly_returns, tickers):
    """inputs() on the last WINDOW months for which the client's products and the market portfolio have data."""
    columns = list(dict.fromkeys(list(tickers) + list(MARKET_PORTFOLIO)))
    return inputs(monthly_returns[columns].dropna().iloc[-WINDOW:], list(tickers))


# Views tilt our recommendation, they do not replace it: the Black-Litterman portfolio stays within a tracking error
# of MAX_ACTIVE_RISK per year from the market equilibrium portfolio (tracking error, LN2). Without this, mean-variance
# reacts to small changes of the expected returns of very similar products: a view on gold moved all the US equities
# into international equities (+1.6% for EFA, +0.6% for SPY, correlation 0.8).
MAX_ACTIVE_RISK = 0.02


def black_litterman_portfolio(pi, cov, gamma, c, views):
    """Mean-variance with the Black-Litterman returns, moved back towards the equilibrium portfolio if its tracking
    error against it is above MAX_ACTIVE_RISK. A mix of two portfolios that respect the limits respects them."""
    equilibrium = mean_variance(pi, cov, gamma, c)
    with_views = mean_variance(posterior_returns(pi, cov, views), cov, gamma, c)
    tracking_error = np.sqrt((with_views - equilibrium) @ cov @ (with_views - equilibrium))
    k = min(1.0, MAX_ACTIVE_RISK / tracking_error) if tracking_error > 0 else 1.0
    return equilibrium + k * (with_views - equilibrium)


def build_portfolios(mu, cov, pi, gamma, c, views, n_sim=100, only=None):
    """The methods, all with the same client constraints (Black-Litterman only if the client has views).
    mu = past average returns, pi = market equilibrium returns. only: list of methods to compute (None = all).
    Returns (weights of the methods that worked, error message of the methods that failed):
    one failing method does not hide the others."""
    views = restrict_views(views, mu.index)  # ignore views on products not chosen
    methods = {
        "Equilibrium": lambda: mean_variance(pi, cov, gamma, c),
        "Minimum variance": lambda: minimum_variance(cov, c),
        "Black-Litterman": lambda: black_litterman_portfolio(pi, cov, gamma, c, views),
        "Mean-variance": lambda: mean_variance(mu, cov, gamma, c),
        "Resampled MV": lambda: resampled(mu, cov, gamma, c, n_sim),
        "Risk parity (ERC)": lambda: risk_parity(cov, c),
    }
    if not views:  # Black-Litterman only for experienced clients who entered views
        del methods["Black-Litterman"]
    if only is not None:
        methods = {name: compute for name, compute in methods.items() if name in only}
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


def gamma_for_limits(forecast, cov, gamma, c, max_loss):
    """Risk aversion actually used: the client's gamma, raised if needed
    1. so that the loss in a bad year stays within his loss limit (max_loss, e.g. -0.10; None = no limit). We ignore
       the expected return (prudent), so the volatility must be at most |max_loss| / 1.645;
    2. so that the maximum in risky assets (horizon, capacity) is not binding. Otherwise the optimiser fills the
       allowed risky budget with the products of highest expected return (REITs, emerging markets) instead of a
       balanced mix: a more prudent gamma gives the same amount of risky assets, but well diversified."""
    if max_loss is not None:
        gamma = gamma_for_max_volatility(forecast, cov, gamma, c, -max_loss / BAD_YEAR)
    if 0.02 <= c["max_equity"] < 1:
        risky = lambda w: sum(v for k, v in class_exposures(w).items() if k != "Bond")
        gamma = gamma_for(forecast, cov, gamma, c, lambda w: risky(w) > c["max_equity"] - 0.01)
    return gamma


# Why we recommend each method (shown to the client in the recommendation box)
WHY = {
    "Equilibrium": "Mean-variance (Markowitz) with the returns the market as a whole expects: the highest expected "
                   "return for the risk you accept (your γ), inside your limits.",
    "Minimum variance": "You may need this money within a year: the goal is to keep your capital, not to make it "
                        "grow. We choose the portfolio with the lowest risk inside your limits; it needs no "
                        "forecast of returns.",
    "Risk parity (ERC)": "Your risk answers contradict each other, so we cannot measure your γ reliably. Risk parity "
                         "needs neither γ nor a forecast: every product brings the same share of the risk.",
    "Black-Litterman": "You entered your own market views: Black-Litterman mixes them with the market equilibrium, "
                       "according to your confidence, then optimises for your γ and your limits. Your views tilt "
                       "the portfolio, but at most 2% tracking error per year away from the equilibrium portfolio.",
}


def recommended_method(profile, portfolios, forecast, cov):
    """The method we recommend to this client (rules at the top of this file)."""
    if profile.preserve_capital:  # safety first, even for a client with views
        return "Minimum variance"
    if "Black-Litterman" in portfolios:  # built only when the client entered views
        return "Black-Litterman"
    erc = portfolios.get("Risk parity (ERC)")
    if profile.inconsistent and erc is not None and (
            profile.max_loss is None or bad_year_loss(erc, forecast, cov) >= profile.max_loss - 1e-4):
        return "Risk parity (ERC)"
    return "Equilibrium"


def dominated_by(portfolios, forecast, cov, method):
    """Methods whose portfolio has a higher expected return AND a lower risk than `method`, with our forecast."""
    ret, vol = portfolio_stats(portfolios[method], forecast, cov)
    better = []
    for other, w in portfolios.items():
        r, v = portfolio_stats(w, forecast, cov)
        if r >= ret - 1e-9 and v <= vol + 1e-9 and (r > ret + 1e-6 or v < vol - 1e-6):
            better.append(other)
    return better


def benchmark_weights(name, tickers, equity_share, currency):
    """Weights of the benchmark chosen by the client. equity_share: equities in the recommended portfolio."""
    if name.startswith("Same equity share"):
        return pd.Series({"SPY": equity_share, BENCHMARK_BOND[currency]: 1 - equity_share})
    if name.startswith("60/40"):
        return pd.Series({"SPY": 0.6, "AGG": 0.4})
    if name.startswith("Equal"):
        return pd.Series(1 / len(tickers), index=tickers)
    if name.startswith("Market"):
        return pd.Series(MARKET_PORTFOLIO)
    return pd.Series({name.split()[1]: 1.0})  # "100% SPY (...)" -> SPY
