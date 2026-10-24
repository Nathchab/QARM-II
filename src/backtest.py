"""
Out-of-sample backtest.

At each rebalancing date we estimate returns and risk on the PAST 60 months only and compute
new weights. Between two rebalancings we hold the portfolio, so the weights drift with the
markets. We never use future information.

Transaction costs, as in lecture LN3:
  - every trade costs `cost` (e.g. 0.10% of the amount traded), paid from the portfolio;
  - Mean-variance and Black-Litterman can be "cost-aware": they solve the LN3 problem with
    transaction costs, so they only trade when it is worth it;
  - the turnover of each rebalancing can be limited (LN3 turnover management).
"""
import numpy as np
import pandas as pd

from src.black_litterman import posterior_returns, restrict_views
from src.optimizer import mean_variance_with_costs
from src.portfolios import WINDOW, build_portfolios

N_SIM = 20  # fewer resampling simulations than on the portfolio page, so the backtest stays fast


def move_towards(old, target, max_turnover):
    """Go from the old weights towards the target, but trade at most max_turnover.
    (A mix of two portfolios that respect the constraints also respects them.)"""
    turnover = (target - old).abs().sum()
    k = min(1, max_turnover / turnover) if turnover > 0 else 1
    return old + k * (target - old)


def run_backtest(returns, rf, tickers, benchmark, gamma, c, views, start, end,
                 rebalance=3, cost=0.001, cost_aware=True, max_turnover=2.0):
    """
    returns: monthly returns of all products; tickers: products of the client.
    benchmark: weights of the benchmark. start, end: dates of the backtest.
    Returns (monthly net returns of each strategy, weights at each rebalancing, turnover at each rebalancing).
    """
    returns = returns[sorted(set(tickers) | set(benchmark.index))].dropna()
    rf = rf.reindex(returns.index).ffill()
    dates = [t for t in range(WINDOW, len(returns)) if start <= returns.index[t] <= end]

    results, weights_history, turnover_history = {}, {}, {}
    weights = {}  # current weights of each strategy (empty = we start in cash)
    for i, t in enumerate(dates):
        date = returns.index[t]
        r = returns.iloc[t]
        traded = {}

        if i % rebalance == 0:  # ---------- rebalancing ----------
            past = returns[tickers].iloc[t - WINDOW:t]
            mu, cov, rf_now = past.mean() * 12, past.cov() * 12, rf.iloc[t - 1] * 12
            targets = build_portfolios(mu, cov, rf_now, gamma, c, views, N_SIM)
            targets["Benchmark"] = benchmark

            if not weights:  # first month: we buy everything with cash
                new = targets
            else:
                limit = max_turnover
                cost_in_optimisation = cost if cost_aware else 0.0
                bl_mu = posterior_returns(cov, rf_now, restrict_views(views, tickers))
                new = {
                    "Mean-variance": mean_variance_with_costs(mu, cov, gamma, c, weights["Mean-variance"].values,
                                                              cost_in_optimisation, limit),
                    "Resampled MV": move_towards(weights["Resampled MV"], targets["Resampled MV"], limit),
                    "Black-Litterman": mean_variance_with_costs(bl_mu, cov, gamma, c, weights["Black-Litterman"].values,
                                                                cost_in_optimisation, limit),
                    "Risk parity (ERC)": move_towards(weights["Risk parity (ERC)"], targets["Risk parity (ERC)"], limit),
                    "Benchmark": benchmark,  # the benchmark always goes back to its weights
                }
            for name, w in new.items():
                traded[name] = (w - weights[name]).abs().sum() if weights else 1.0
                weights_history.setdefault(name, {})[date] = w
                turnover_history.setdefault(name, {})[date] = traded[name]
            weights = new

        # ---------- one month of returns ----------
        month = {}
        for name, w in weights.items():
            month[name] = r[w.index] @ w - cost * traded.get(name, 0.0)  # return minus trading costs
            weights[name] = w * (1 + r[w.index]) / (1 + r[w.index] @ w)  # the weights drift
        results[date] = month

    weights_history = {m: pd.DataFrame(h).T for m, h in weights_history.items()}
    return pd.DataFrame(results).T, weights_history, pd.DataFrame(turnover_history)


def performance(r, rf, benchmark, turnover, cost):
    """Usual performance measures from monthly returns (rf and benchmark: monthly too)."""
    rf = rf.reindex(r.index).ffill()
    value = (1 + r).cumprod()
    years = len(r) / 12
    return {
        "Annual return": value.iloc[-1] ** (1 / years) - 1,
        "Volatility": r.std() * np.sqrt(12),
        "Sharpe ratio": (r - rf).mean() / r.std() * np.sqrt(12),
        "Max drawdown": (value / value.cummax() - 1).min(),
        "Beta vs benchmark": r.cov(benchmark) / benchmark.var(),          # LN2
        "Tracking error": (r - benchmark).std() * np.sqrt(12),            # LN2
        "Turnover per year": turnover.iloc[1:].sum() / years,             # without the first purchase
        "Costs paid per year": cost * turnover.sum() / years,
    }
