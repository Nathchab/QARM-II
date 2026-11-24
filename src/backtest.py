"""
Out-of-sample backtest.

At each rebalancing date we estimate returns and risk on the PAST 60 months only and compute
new weights. Between two rebalancings we hold the portfolio, so the weights drift with the
markets. We never use future information.

Our management fee (MANAGEMENT_FEE per year) is taken every month from our portfolios, not from
the benchmark: the client would not pay it if he bought the benchmark himself.

Transaction costs, as in lecture LN3:
  - every trade costs `cost` (e.g. 0.10% of the amount traded), paid from the portfolio in the month of the trade
    (the first month pays for buying the whole portfolio);
  - the mean-variance methods (equilibrium, past returns, Black-Litterman) can be "cost-aware": they solve the LN3 problem with
    transaction costs, so they only trade when it is worth it;
  - the turnover of each rebalancing can be limited (LN3 turnover management).
"""
import numpy as np
import pandas as pd

from src.black_litterman import MARKET_PORTFOLIO, posterior_returns, prior_returns, restrict_views
from src.optimizer import mean_variance_with_costs
from src.planning import MANAGEMENT_FEE
from src.portfolios import WINDOW, build_portfolios, gamma_for_limits, inputs

N_SIM = 20  # fewer resampling simulations than on the portfolio page, so the backtest stays fast


def move_towards(old, target, max_turnover):
    """Go from the old weights towards the target, but trade at most max_turnover.
    (A mix of two portfolios that respect the constraints also respects them.)"""
    turnover = (target - old).abs().sum()
    k = min(1, max_turnover / turnover) if turnover > 0 else 1
    return old + k * (target - old)


def run_backtest(returns, rf, tickers, benchmark, gamma, c, views, start, end,
                 rebalance=3, cost=0.001, cost_aware=True, max_turnover=2.0, max_loss=None):
    """
    returns: monthly returns of all products; tickers: products of the client.
    benchmark: weights of the benchmark. start, end: dates of the backtest.
    Returns (monthly returns of each strategy after transaction costs and fee, weights at each rebalancing,
    turnover at each rebalancing, transaction costs paid each month).
    """
    returns = returns[sorted(set(tickers) | set(benchmark.index) | set(MARKET_PORTFOLIO))].dropna()
    rf = rf.reindex(returns.index).ffill()
    dates = [t for t in range(WINDOW, len(returns)) if start <= returns.index[t] <= end]

    results, weights_history, turnover_history, costs_paid = {}, {}, {}, {}
    weights = {}  # current weights of each strategy (empty = we start in cash)
    for i, t in enumerate(dates):
        date = returns.index[t]
        r = returns.iloc[t]
        traded = {}

        if i % rebalance == 0:  # ---------- rebalancing ----------
            mu, cov, market_cov = inputs(returns.iloc[t - WINDOW:t], tickers)
            pi = prior_returns(market_cov, rf.iloc[t - 1] * 12)
            bl_mu = posterior_returns(pi, cov, restrict_views(views, tickers))
            g = gamma_for_limits(bl_mu, cov, gamma, c, max_loss)  # same loss limit as on the portfolio page
            targets, errors = build_portfolios(mu, cov, pi, g, c, views, N_SIM)
            if errors:  # a method failed on this date: we stop and show its message
                raise RuntimeError(list(errors.values())[0])
            targets["Benchmark"] = benchmark

            if not weights:  # first month: we buy everything with cash
                new = targets
            else:
                limit = max_turnover
                cost_in_optimisation = cost if cost_aware else 0.0
                forecasts = {"Equilibrium": pi, "Minimum variance": 0 * mu, "Mean-variance": mu,
                             "Black-Litterman": bl_mu}  # minimum variance = mean-variance with mu = 0
                new = {}
                for name in targets:
                    if name == "Benchmark":  # the benchmark always goes back to its weights
                        new[name] = benchmark
                    elif name in forecasts:  # mean-variance methods: LN3 problem with costs and turnover limit
                        new[name] = mean_variance_with_costs(forecasts[name], cov, g, c, weights[name].values,
                                                             cost_in_optimisation, limit)
                    else:  # resampled MV and risk parity: move towards the new target
                        new[name] = move_towards(weights[name], targets[name], limit)
            for name, w in new.items():
                traded[name] = (w - weights[name]).abs().sum() if weights else 1.0
                weights_history.setdefault(name, {})[date] = w
                turnover_history.setdefault(name, {})[date] = traded[name]
            weights = new

        # ---------- one month of returns ----------
        month, paid = {}, {}
        for name, w in weights.items():
            fee = 0.0 if name == "Benchmark" else MANAGEMENT_FEE / 12
            paid[name] = cost * traded.get(name, 0.0)  # only in the months when the portfolio trades
            month[name] = r[w.index] @ w - paid[name] - fee  # return minus transaction costs and fee
            weights[name] = w * (1 + r[w.index]) / (1 + r[w.index] @ w)  # the weights drift
        results[date], costs_paid[date] = month, paid

    weights_history = {m: pd.DataFrame(h).T for m, h in weights_history.items()}
    return pd.DataFrame(results).T, weights_history, pd.DataFrame(turnover_history), pd.DataFrame(costs_paid).T


def performance(r, rf, benchmark, turnover, costs, fee):
    """Usual performance measures from monthly returns r, after transaction costs and fee (rf and benchmark:
    monthly too). costs: transaction costs paid each month; fee: management fee per year (0 for the benchmark).
    The return before costs and fee adds them back: r + costs + fee / 12."""
    rf = rf.reindex(r.index).ffill()
    value = (1 + r).cumprod()
    years = len(r) / 12
    annual = lambda monthly: (1 + monthly).prod() ** (1 / years) - 1
    return {
        "Annual return before costs and fee": annual(r + costs + fee / 12),
        "Transaction costs per year": costs.sum() / years,
        "Management fee per year": fee,
        "Annual return after costs and fee": value.iloc[-1] ** (1 / years) - 1,
        "Volatility": r.std() * np.sqrt(12),
        "Sharpe ratio": (r - rf).mean() / r.std() * np.sqrt(12),
        "Max drawdown": (value / value.cummax() - 1).min(),
        "Beta vs benchmark": r.cov(benchmark) / benchmark.var(),          # LN2
        "Tracking error": (r - benchmark).std() * np.sqrt(12),            # LN2
        "Turnover per year": turnover.iloc[1:].sum() / years,             # without the first purchase
    }
