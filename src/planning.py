"""
Robo-advisor planning tools: wealth projection with a savings plan, savings goal, order list and client report.

Projection: we simulate many possible futures (Monte Carlo, as for the resampling of lab LN3).
Each month the portfolio earns a random return ~ Normal(mu/12, vol/sqrt(12)), net of our management fee,
then the client adds his monthly contribution.
"""
from datetime import date

import numpy as np
import pandas as pd

from src.data import ASSETS

MANAGEMENT_FEE = 0.005  # our robo-advisor fee: 0.50% of the portfolio per year
# QD1 (when the client needs the money) -> default horizon of the projection, in years
HORIZON_YEARS = {"a": 1, "b": 3, "c": 5, "d": 10, "e": 15}


def simulate_wealth(mu, vol, years, initial, monthly, n_paths=5000):
    """Simulate the client's wealth month by month.
    mu, vol: annual expected return (already net of fees) and volatility of the portfolio.
    Returns (wealth: n_paths x (months+1), growth: value of 1 invested at the start, without contributions)."""
    np.random.seed(42)  # same simulations every time, so the page always shows the same numbers
    months = int(years * 12)
    returns = np.random.normal(mu / 12, vol / np.sqrt(12), size=(n_paths, months))
    wealth = np.zeros((n_paths, months + 1))
    wealth[:, 0] = initial
    for t in range(months):
        wealth[:, t + 1] = wealth[:, t] * (1 + returns[:, t]) + monthly
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


def client_report(profile, c, method, weights, exp_return, vol, currency, plan, orders, cash):
    """Plain-text summary of the advice, that the client can download and keep."""
    pct = lambda x: f"{x:.1%}"
    lines = [
        "ROBO-ADVISOR – CLIENT REPORT",
        f"Date: {date.today():%d %B %Y}    Currency: {currency}",
        "",
        "1. YOUR PROFILE",
        f"Risk tolerance: {profile.rt:.1f}/100 ({profile.label}), risk aversion gamma = {profile.gamma:.2f}",
        f"Financial capacity: {profile.capacity_label} ({profile.capacity_score}/9)",
        f"Limits: equities max {pct(c['max_equity'])}, bonds min {pct(c['min_bonds'])}, short-term Treasuries min "
        f"{pct(c.get('min_short_term', 0))}, gold and commodities max "
        f"{pct(c['max_other'])}, crypto max {pct(c['max_crypto'])}, max {pct(c['max_weight_per_asset'])} per product",
        "",
        f"2. YOUR PORTFOLIO ({method})",
        *[f"  {t:8s} {ASSETS[t][0]:40s} {pct(w):>7s}" for t, w in weights.items() if w >= 0.0005],
        f"Expected return {pct(exp_return)} per year before our fee of {pct(MANAGEMENT_FEE)}, volatility {pct(vol)}",
        "",
        "3. YOUR PLAN",
        f"Initial amount {plan['initial']:,.0f}, monthly contribution {plan['monthly']:,.0f}, "
        f"horizon {plan['years']} years (total paid in {plan['paid_in']:,.0f})",
        f"Wealth at the end: pessimistic (5%) {plan['p5']:,.0f}, median {plan['p50']:,.0f}, "
        f"optimistic (95%) {plan['p95']:,.0f}",
        f"Probability of reaching your goal of {plan['goal']:,.0f}: {pct(plan['prob_goal'])}",
        f"Probability of a fall of more than {pct(plan['loss_limit'])} at some point: {pct(plan['prob_loss'])}",
        "",
        "4. ORDERS TO PLACE",
        *[f"  {row.Product:50s} {row.Quantity:>10.4g} units  {row.Amount:>12,.0f}" for row in orders.itertuples()],
        f"  Cash left: {cash:,.0f}",
        "",
        "5. POINTS TO REVIEW",
        *([f"  - {w}" for w in profile.warnings] or ["  None."]),
        "",
        "Academic project (QARM II, HEC Lausanne). This is not investment advice.",
    ]
    return "\n".join(lines)
