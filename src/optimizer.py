"""
Portfolio optimisation with the client's constraints.
The structure follows our labs (QP function, LinearConstraint, scipy minimize).

All inputs are annual and are pandas objects:
mu = expected returns (Series), cov = covariance matrix (DataFrame).
c = the client's constraints: max_equity, min_bonds, min_short_term, max_other, max_crypto, max_weight_per_asset,
    and (optional) min_home_equity = minimum share of the equities in the client's region c["region"].
"""
import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, linprog, minimize

from src.data import ASSETS, HOME_REGIONS, REGION_SHARES, REGIONS, SATELLITE_MAX, SATELLITES, SHORT_TERM

CLASSES = ["Equity", "Bond", "Other", "Crypto"]
CASH_LIKE = list(SHORT_TERM.values())  # SHY and the Swiss government bonds 0-3 years


# ---------- The client's constraints ----------
def class_rows(tickers, extra_variables=0):
    """For each asset class, a vector of 0/1: (vector @ x) is the weight of this class in the portfolio x.
    extra_variables: number of extra variables after the weights (used with transaction costs)."""
    return {k: np.array([float(ASSETS[t][1] == k) for t in tickers] + [0.0] * extra_variables) for k in CLASSES}


def short_term_row(tickers, extra_variables=0):
    """(vector @ x) = weight in short-term government bonds (SHY or Swiss 0-3 years)."""
    return np.array([float(t in CASH_LIKE) for t in tickers] + [0.0] * extra_variables)


def home_row(tickers, c, extra_variables=0):
    """(vector @ x) >= 0  <=>  at least min_home_equity of the equities are in the client's region.
    Each equity product counts with its share in the region (ACWI is 2% Swiss): sum (share_i - min) x_i >= 0."""
    home = HOME_REGIONS.get(c.get("region"), [])
    return np.array([sum(REGION_SHARES[t].get(r, 0) for r in home) - c.get("min_home_equity", 0)
                     if ASSETS[t][1] == "Equity" else 0.0 for t in tickers] + [0.0] * extra_variables)


def lacks_home_equity(tickers, c):
    """True if the client has equity products, but none mainly in his region: the regional rule then forces
    his equities to 0%."""
    equity_rows = [r for t, r in zip(tickers, home_row(tickers, c)) if ASSETS[t][1] == "Equity"]
    return c.get("min_home_equity", 0) > 0 and bool(equity_rows) and max(equity_rows) < 0


def limits(tickers, c, extra_variables=0):
    """The client's limits as a list of (name, row, minimum, maximum): minimum <= row @ x <= maximum."""
    rows, inf = class_rows(tickers, extra_variables), np.inf
    rules = [("maximum in equities", rows["Equity"], -inf, c["max_equity"]),
             ("minimum in bonds", rows["Bond"], c["min_bonds"], inf),
             ("maximum in gold and commodities", rows["Other"], -inf, c["max_other"]),
             ("maximum in crypto", rows["Crypto"], -inf, c["max_crypto"])]
    if c.get("min_short_term", 0) > 0:  # money that may be needed soon
        rules.append(("minimum in short-term government bonds", short_term_row(tickers, extra_variables),
                      c["min_short_term"], inf))
    if c.get("min_home_equity", 0) > 0:  # region chosen in QF3
        rules.append((f"minimum share of your equities in {c['region']}", home_row(tickers, c, extra_variables),
                      0, inf))
    return rules


def binding_limits(weights, c):
    """Names of the limits on which the portfolio sits exactly. Each limit has one bound (a minimum or a maximum).
    A bound of 0% (no bonds required, no crypto allowed) is not worth mentioning; the regional rule is written
    'row @ x >= 0', so it is kept."""
    found = []
    for name, row, low, high in limits(list(weights.index), c):
        bound = low if np.isfinite(low) else high
        if abs(row @ weights.values - bound) <= 1e-6 and (bound != 0 or name.startswith("minimum share")):
            found.append(name)
    return found


def client_constraints(tickers, c, extra_variables=0):
    """LN3 'asset class constraints' written as LinearConstraint (as in the labs)."""
    return [LinearConstraint(row, lb=low, ub=high) for name, row, low, high in limits(tickers, c, extra_variables)]


def weight_bounds(tickers, c):
    """No short selling and a maximum weight per product: 0 <= x_i <= max.
    Short-term government bonds are almost cash, so they have no maximum; satellites have at most SATELLITE_MAX."""
    cap = c["max_weight_per_asset"]
    return [(0, 1.0 if t in CASH_LIKE else min(cap, SATELLITE_MAX) if t in SATELLITES else cap) for t in tickers]


def region_exposures(weights):
    """Weight of each region in a portfolio, looking through the ETFs (ACWI is 67% North America, ...)."""
    return {r: sum(w * REGION_SHARES[t].get(r, 0) for t, w in weights.items()) for r in REGIONS}


def class_exposures(weights):
    """Weight of each asset class in a portfolio (weights = Series indexed by ticker)."""
    return {k: sum(w for t, w in weights.items() if ASSETS[t][1] == k) for k in CLASSES}


def respects_limits(x, tickers, c, tolerance=1e-9):
    """True if the weights x are long-only, fully invested and inside the client's limits."""
    return bool(abs(np.sum(x) - 1) <= tolerance
                and all(lo - tolerance <= w <= hi + tolerance for w, (lo, hi) in zip(x, weight_bounds(tickers, c)))
                and all(low - tolerance <= row @ x <= high + tolerance for name, row, low, high in limits(tickers, c)))


def feasible_portfolio(tickers, c):
    """Any portfolio that respects the limits, or None if there is none.
    Found with a linear programme written as in LN3: S x <= T, sum(x) = 1, 0 <= x <= max weight."""
    rules = limits(tickers, c)
    upper = [(row, high) for name, row, low, high in rules if high < np.inf]
    lower = [(-row, -low) for name, row, low, high in rules if low > -np.inf]  # low <= row @ x <=> -row @ x <= -low
    S, T = zip(*upper + lower)
    result = linprog(np.zeros(len(tickers)), A_ub=S, b_ub=T, A_eq=[np.ones(len(tickers))], b_eq=[1.0],
                     bounds=weight_bounds(tickers, c))
    return result.x if result.success else None


def check_constraints(tickers, c):
    """Return a list of problems if no portfolio can respect the constraints (empty list = feasible)."""
    if len(tickers) < 2:
        return ["Choose at least 2 products."]
    if feasible_portfolio(tickers, c) is not None:
        return []
    # Explain why: how much can each asset class hold at most?
    room = {k: sum(hi for t, (lo, hi) in zip(tickers, weight_bounds(tickers, c)) if ASSETS[t][1] == k) for k in CLASSES}
    problems = []
    if c.get("min_short_term", 0) > 0 and not any(t in CASH_LIKE for t in tickers):
        problems.append(f"You need at least {c['min_short_term']:.0%} in short-term government bonds: keep "
                        f"{' or '.join(CASH_LIKE)}.")
    if lacks_home_equity(tickers, c):
        problems.append(f"At least {c['min_home_equity']:.0%} of your equities must be in {c['region']}: add an "
                        "equity product of this region.")
    if room["Bond"] < c["min_bonds"] - 1e-9:
        problems.append(f"Your bond products can hold at most {room['Bond']:.0%}, below the minimum of "
                        f"{c['min_bonds']:.0%} in bonds: add bond products.")
    total = (min(room["Equity"], c["max_equity"]) + room["Bond"] + min(room["Other"], c["max_other"])
             + min(room["Crypto"], c["max_crypto"]))
    if total < 1 - 1e-9:
        problems.append(f"With these products and limits you can invest only {total:.0%}: "
                        "add products (bonds for example) or raise the maximum per product.")
    return problems or ["These products and limits cannot be satisfied together."]


def starting_point(tickers, c):
    """Where the optimiser starts: equal weights if they respect the limits, otherwise another portfolio that does."""
    equal = np.ones(len(tickers)) / len(tickers)
    start = equal if respects_limits(equal, tickers, c, 1e-12) else feasible_portfolio(tickers, c)
    if start is None:
        raise ValueError("These products and limits cannot be satisfied together.")
    return start


def clean(x, tickers):
    """Remove tiny numerical errors (e.g. -0.0001%) and make the weights sum to 1."""
    w = np.clip(x, 0, 1)
    return pd.Series(w / w.sum(), index=tickers)


# ---------- Mean-variance ----------
def QP(x, sigma, mu, gamma):
    """Lab LN1: 0.5 x'Sigma x - gamma x'mu.
    Careful: here gamma is the risk TOLERANCE = 1 / risk aversion of the client."""
    return 0.5 * x.T @ sigma @ x - gamma * x.T @ mu


def QP_gradient(x, sigma, mu, gamma):
    """Derivative of QP with respect to x: Sigma x - gamma mu. It makes the solver faster and more precise."""
    return sigma @ x - gamma * mu


def mean_variance(mu, cov, risk_aversion, c):
    """Markowitz portfolio of a client with this risk aversion (gamma from the questionnaire)."""
    tickers = list(mu.index)
    n = len(tickers)
    gamma = 1 / risk_aversion  # the lab's gamma
    # We divide the whole objective by its largest number: the solution is the same,
    # but the solver works with numbers around 1 whatever the units of the inputs.
    scale = max(np.abs(cov.values).max(), gamma * np.abs(mu.values).max())
    rules = [LinearConstraint(np.ones(n), lb=1, ub=1)] + client_constraints(tickers, c)  # 100% invested
    res = minimize(QP, starting_point(tickers, c), args=(cov.values / scale, mu.values / scale, gamma),
                   jac=QP_gradient, bounds=weight_bounds(tickers, c), constraints=rules, method="SLSQP",
                   options={"ftol": 1e-12, "maxiter": 2000})
    if not res.success:  # we prefer an error message to a wrong portfolio
        raise RuntimeError(f"Mean-variance SLSQP failed (status {res.status}): {res.message}")
    return pd.Series(res.x, index=tickers)


def minimum_variance(cov, c):
    """Portfolio with the lowest volatility inside the client's limits: no expected returns needed
    (mean-variance with mu = 0)."""
    return mean_variance(pd.Series(0.0, index=cov.index), cov, 1.0, c)


def mean_variance_with_costs(mu, cov, risk_aversion, c, x0, cost, max_turnover=2.0):
    """
    LN3 'transaction cost management' and 'turnover management'.
    We start from the current portfolio x0 and look for the new portfolio x:
        x = x0 + dx_plus - dx_minus   (dx_plus = what we buy, dx_minus = what we sell)
    Variables: X = (x, dx_minus, dx_plus), 3n numbers, as in the lecture.
        min 0.5 x'Sigma x - gamma (x'mu - c'dx_minus - c'dx_plus)
        u.c. sum(x) + c'dx_minus + c'dx_plus = 1     (the costs are paid from the portfolio)
             x + dx_minus - dx_plus = x0
             sum(dx_minus) + sum(dx_plus) <= max turnover
             client constraints on x, 0 <= X <= 1
    So we only trade when the gain in expected utility is larger than the cost.
    """
    tickers = list(mu.index)
    n = len(tickers)
    gamma = 1 / risk_aversion
    costs = np.full(n, cost)

    # Same scaling, gradient and precision as mean_variance: without them SLSQP stops after a few steps, close to
    # x0, and the portfolio trades much less than it should
    scale = max(np.abs(cov.values).max(), gamma * np.abs(mu.values).max())
    Q = np.zeros((3 * n, 3 * n))
    Q[:n, :n] = cov.values / scale                                    # only x has risk
    R = gamma * np.concatenate([mu.values, -costs, -costs]) / scale   # expected return minus costs
    objective = lambda X: 0.5 * X @ Q @ X - X @ R
    gradient = lambda X: Q @ X - R

    I = np.eye(n)
    rules = [
        LinearConstraint(np.concatenate([np.ones(n), costs, costs]), lb=1, ub=1),  # budget
        LinearConstraint(np.hstack([I, I, -I]), lb=x0, ub=x0),                    # link with x0
        LinearConstraint(np.concatenate([np.zeros(n), np.ones(2 * n)]), ub=max_turnover),  # turnover
    ] + client_constraints(tickers, c, extra_variables=2 * n)
    bounds = weight_bounds(tickers, c) + [(0, 1)] * (2 * n)  # 0 <= dx_minus, dx_plus <= 1
    start = np.concatenate([x0, np.zeros(2 * n)])
    res = minimize(objective, start, jac=gradient, bounds=bounds, constraints=rules, method="SLSQP",
                   options={"ftol": 1e-12, "maxiter": 2000})
    if not res.success:  # we prefer an error message to a wrong portfolio
        raise RuntimeError(f"Mean-variance with costs SLSQP failed (status {res.status}): {res.message}")
    return clean(res.x[:n], tickers)


def resampled(mu, cov, risk_aversion, c, n_sim=100, n_months=60):
    """Lab LN3, resampling of Michaud (2007):
    simulate returns, re-estimate mu and cov, optimise, and average the weights."""
    np.random.seed(42)  # same simulations every time, so the app always shows the same result
    all_weights = []
    for i in range(n_sim):
        simulated_r = np.random.multivariate_normal(mu / 12, cov / 12, n_months)  # monthly returns
        simulated_mu = pd.Series(simulated_r.mean(axis=0) * 12, index=mu.index)
        simulated_cov = pd.DataFrame(np.cov(simulated_r, rowvar=False) * 12, index=mu.index, columns=mu.index)
        all_weights.append(mean_variance(simulated_mu, simulated_cov, risk_aversion, c))
    return pd.concat(all_weights, axis=1).mean(axis=1)


def frontier(mu, cov, c, gammas=np.geomspace(0.25, 200, 40)):
    """Efficient frontier and composition map (lab LN3): optimal portfolio for many risk aversions."""
    rows = []
    for g in gammas:
        w = mean_variance(mu, cov, g, c)
        ret, vol = portfolio_stats(w, mu, cov)
        rows.append({"Gamma": g, "Return": ret, "Volatility": vol, **w})
    return pd.DataFrame(rows)


def gamma_for_max_volatility(mu, cov, gamma, c, max_vol):
    """Lab LN1 'targeting a particular volatility': the smallest risk aversion (at least gamma) whose
    mean-variance portfolio has a volatility of at most max_vol. Found by bisection: the volatility of the
    optimal portfolio goes down when gamma goes up. If even gamma = 1000 is too risky, we return 1000
    (the safest portfolio we can build with these products)."""
    volatility = lambda g: portfolio_stats(mean_variance(mu, cov, g, c), mu, cov)[1]
    if max_vol is None or volatility(gamma) <= max_vol:
        return gamma
    low, high = gamma, 1000.0
    if volatility(high) > max_vol:
        return high
    for i in range(15):
        middle = np.sqrt(low * high)
        if volatility(middle) > max_vol:
            low = middle
        else:
            high = middle
    return high


# ---------- Risk parity ----------
def risk_contributions(weights, cov):
    """LN5: share of the portfolio risk that comes from each asset, x_i (Sigma x)_i / x'Sigma x (sums to 100%)."""
    w, sigma = np.asarray(weights), np.asarray(cov)
    return w * (sigma @ w) / (w @ sigma @ w)


def risk_parity(cov, c):
    """Equal Risk Contribution (ERC, LN5): every asset brings the same share of risk.
    With the client's constraints we get as close as possible to equal shares."""
    tickers = list(cov.index)
    n = len(tickers)
    sigma = cov.values
    objective = lambda w: 1000 * ((risk_contributions(w, sigma) - 1 / n) ** 2).sum()
    rules = [LinearConstraint(np.ones(n), lb=1, ub=1)] + client_constraints(tickers, c)
    res = minimize(objective, np.ones(n) / n, bounds=weight_bounds(tickers, c), constraints=rules,
                   method="SLSQP")
    weights = clean(res.x, tickers)
    # When the limits leave almost no room, SLSQP can stop without confirming that it converged.
    # We then keep its last portfolio, but only if it respects the client's limits.
    if not res.success and not respects_limits(weights.values, tickers, c, 1e-4):
        raise RuntimeError(f"Risk parity SLSQP failed: {res.message}")
    return weights


# ---------- Measures ----------
def portfolio_stats(weights, mu, cov):
    """Expected return and volatility of a portfolio (annual)."""
    w = np.asarray(weights)
    return w @ mu.values, np.sqrt(w @ cov.values @ w)


def beta(portfolio_returns, market_returns):
    """Lab LN2, covariance method: beta = cov(R_p, R_m) / var(R_m)."""
    return portfolio_returns.cov(market_returns) / market_returns.var()


def tracking_error(portfolio_returns, benchmark_returns):
    """LN2: volatility of the difference between the portfolio and the benchmark (annual)."""
    return (portfolio_returns - benchmark_returns).std() * np.sqrt(12)
