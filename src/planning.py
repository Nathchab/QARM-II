"""
Robo-advisor planning tools: wealth projection with a savings plan, savings goal and order list
(the PDF report is in report.py).

Projection: we simulate many possible futures (Monte Carlo, as for the resampling of lab LN3).
Each month the portfolio earns a random return ~ Normal(mu/12, vol/sqrt(12)), net of our management fee,
and the client adds his contribution every 1, 3, 6 or 12 months.
"""
import numpy as np
import pandas as pd

from src.data import ASSETS

MANAGEMENT_FEE = 0.005  # our robo-advisor fee: 0.50% of the portfolio per year
TRADING_COST = 0.001    # default transaction cost: 10 bp of every amount bought or sold (plan and backtest)
# QD1 (when the client needs the money) -> default horizon of the projection, in years
HORIZON_YEARS = {"a": 1, "b": 3, "c": 5, "d": 10, "e": 15}
# How often the client adds money: name -> (number of months between two contributions, words for the texts)
FREQUENCIES = {"Monthly": (1, "per month"), "Quarterly": (3, "per quarter"), "Semi-annual": (6, "every six months"),
               "Annual": (12, "per year")}


def paid_in(initial, contribution, every, months):
    """Total paid in after `months` months: the initial amount plus one contribution every `every` months."""
    return initial + contribution * (np.asarray(months) // every)


def simulate_wealth(mu, vol, years, initial, contribution, every=1, cost=0., n_paths=5000):
    """Simulate the client's wealth month by month.
    mu, vol: annual expected return (already net of our fee) and volatility of the portfolio.
    contribution: amount added at the end of every `every` months (1 = monthly, 3 = quarterly...).
    cost: transaction cost paid on each purchase (the initial amount and every contribution are invested minus it).
    Returns (wealth: n_paths x (months+1), growth: value of 1 invested at the start, without contributions)."""
    np.random.seed(42)  # same simulations every time, so the page always shows the same numbers
    months = int(years * 12)
    returns = np.random.normal(mu / 12, vol / np.sqrt(12), size=(n_paths, months))
    wealth = np.zeros((n_paths, months + 1))
    wealth[:, 0] = initial * (1 - cost)
    for t in range(months):
        added = contribution * (1 - cost) if (t + 1) % every == 0 else 0
        wealth[:, t + 1] = wealth[:, t] * (1 + returns[:, t]) + added
    growth = np.hstack([np.ones((n_paths, 1)), np.cumprod(1 + returns, axis=1)])
    return wealth, growth


def worst_drawdowns(growth):
    """Largest fall from a previous peak in each simulated future (e.g. -0.25 = the portfolio lost 25% at worst)."""
    peaks = np.maximum.accumulate(growth, axis=1)
    return (growth / peaks - 1).min(axis=1)


def order_list(weights, last_prices, amount):
    """How many units of each product to buy with `amount`. ETFs are bought in whole shares,
    crypto in fractions (0.0001). Returns (table of orders, cash left)."""
    rows = []
    for ticker, weight in weights.items():
        if weight < 0.0005:  # products with (almost) no weight are not bought
            continue
        step = 0.0001 if ASSETS[ticker][1] == "Crypto" else 1
        quantity = np.floor(amount * weight / last_prices[ticker] / step) * step
        rows.append({"Product": f"{ticker} – {ASSETS[ticker][0]}", "Target weight": weight,
                     "Last price": last_prices[ticker], "Quantity": quantity,
                     "Amount": quantity * last_prices[ticker]})
    orders = pd.DataFrame(rows)
    return orders, amount - orders["Amount"].sum()
