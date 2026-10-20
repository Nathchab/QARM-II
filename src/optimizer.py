"""
Portfolio optimisation with the client's constraints.
The structure follows our labs (QP function, LinearConstraint, scipy minimize).

All inputs are annual and are pandas objects:
mu = expected returns (Series), cov = covariance matrix (DataFrame).
c = the client's constraints: max_equity, min_bonds, max_other, max_crypto, max_weight_per_asset.
"""
import numpy as np
import pandas as pd
from numbers import Real
from scipy.optimize import LinearConstraint, linprog, minimize

from src.data import ASSETS


def QP(x, sigma, mu, gamma):
    """Lab LN1: 0.5 x'Sigma x - gamma x'mu.
    Careful: here gamma is the risk TOLERANCE = 1 / risk aversion of the client."""
    return 0.5 * x.T @ sigma @ x - gamma * x.T @ mu


def client_constraints(tickers, c, extra_variables=0):
    """LN3 'asset class constraints' written as LinearConstraint (as in the labs).
    extra_variables: number of extra variables after the weights (used with transaction costs)."""
    classes = np.array([ASSETS[t][1] for t in tickers])
    pad = np.zeros(extra_variables)
    equity = np.concatenate([(classes == "Equity").astype(float), pad])
    bonds = np.concatenate([(classes == "Bond").astype(float), pad])
    other = np.concatenate([(classes == "Other").astype(float), pad])
    crypto = np.concatenate([(classes == "Crypto").astype(float), pad])
    return [
        LinearConstraint(equity, ub=c["max_equity"]),   # maximum in equities
        LinearConstraint(bonds, lb=c["min_bonds"]),     # minimum in bonds
        LinearConstraint(other, ub=c["max_other"]),     # maximum in gold and commodities
        LinearConstraint(crypto, ub=c["max_crypto"]),   # maximum in crypto
    ]


def weight_bounds(tickers, c):
    """No short selling and maximum weight per product: 0 <= x_i <= max."""
    return [(0, c["max_weight_per_asset"])] * len(tickers)


def check_constraints(tickers, c):
    """Return a list of problems if no portfolio can respect the constraints (empty list = feasible).
    Exact test: a linear programme looks for one long-only, fully invested portfolio within all limits."""
    if len(tickers) < 2:
        return ["Choose at least 2 products."]
    classes = np.array([ASSETS[t][1] for t in tickers])
    rules = np.array([classes == "Equity", -(classes == "Bond").astype(float),
                      classes == "Other", classes == "Crypto"], dtype=float)
    result = linprog(np.zeros(len(tickers)), A_ub=rules,
                     b_ub=[c["max_equity"], -c["min_bonds"], c["max_other"], c["max_crypto"]],
                     A_eq=np.ones((1, len(tickers))), b_eq=[1.0],
                     bounds=weight_bounds(tickers, c), method="highs")
    if result.success:
        return []
    w = c["max_weight_per_asset"]
    room = {k: float((classes == k).sum()) * w for k in ["Equity", "Bond", "Other", "Crypto"]}
    problems = []
    if room["Bond"] < c["min_bonds"] - 1e-9:
        problems.append(f"Your bond products can hold at most {room['Bond']:.0%}, below the minimum of "
                        f"{c['min_bonds']:.0%} in bonds: add bond products.")
    total = (min(room["Equity"], c["max_equity"]) + room["Bond"] + min(room["Other"], c["max_other"])
             + min(room["Crypto"], c["max_crypto"]))
    if total < 1 - 1e-9:
        problems.append(f"With these products and limits you can invest only {total:.0%}: "
                        "add products (bonds for example) or raise the maximum per product.")
    return problems or ["These products and limits cannot be satisfied together."]


def clean(x, tickers):
    """Remove tiny numerical errors (e.g. -0.0001%) and make the weights sum to 1."""
    w = np.clip(x, 0, 1)
    return pd.Series(w / w.sum(), index=tickers)


def _mv_coefficients(mu, cov, risk_aversion):
    """Scale the SAME quadratic by a positive scalar; never regularize covariance.

    The two stages avoid forming mu / risk_aversion or a potentially overflowing
    scale factor in original units. Joint scaling of mu and cov cancels out.
    Symmetrization preserves the quadratic form and its exact gradient.
    """
    if (not isinstance(risk_aversion, Real) or isinstance(risk_aversion, (bool, np.bool_))
            or not np.isfinite(risk_aversion) or risk_aversion <= 0):
        raise ValueError("Mean-variance risk_aversion must be finite and strictly positive.")
    if not isinstance(mu, pd.Series) or not isinstance(cov, pd.DataFrame):
        raise ValueError("Mean-variance expects a return Series and covariance DataFrame.")
    if (mu.empty or not mu.index.is_unique or not cov.index.equals(mu.index)
            or not cov.columns.equals(mu.index)):
        raise ValueError("Mean-variance covariance rows/columns must match the unique return index in order.")
    returns, sigma = mu.to_numpy(dtype=float), cov.to_numpy(dtype=float)
    if not np.isfinite(returns).all() or not np.isfinite(sigma).all():
        raise ValueError("Mean-variance returns and covariance must contain only finite numbers.")
    data_scale = max(np.abs(returns).max(), np.abs(sigma).max())
    if data_scale == 0:
        data_scale = 1.0  # identically zero objective; every feasible point is optimal
    returns, sigma = returns / data_scale, sigma / data_scale
    cov_scale = np.abs(sigma).max()
    if cov_scale > 0:
        unit_cov = sigma / cov_scale
        if np.max(np.abs(unit_cov-unit_cov.T)) > 1e-12:
            raise ValueError("Mean-variance covariance must be symmetric.")
        if np.linalg.eigvalsh(0.5*unit_cov + 0.5*unit_cov.T).min() < -1e-12:
            raise ValueError("Mean-variance covariance must be positive semidefinite; no regularization is applied.")
    sigma = 0.5*sigma + 0.5*sigma.T
    if risk_aversion <= 1:
        sigma = sigma * risk_aversion  # positive multiplier g / data_scale
    else:
        returns = returns / risk_aversion  # positive multiplier 1 / data_scale
    coefficient_scale = max(np.abs(sigma).max(), np.abs(returns).max())
    if coefficient_scale == 0:
        if np.any(mu.to_numpy()) or np.any(cov.to_numpy()):
            raise ValueError("Mean-variance input scale is too extreme for floating-point optimization.")
        coefficient_scale = 1.0
    return sigma / coefficient_scale, returns / coefficient_scale


def _mv_linear_system(rules):
    """Express the existing LinearConstraints for a feasibility LP."""
    equality, eq_rhs, inequality, ineq_rhs = [], [], [], []
    for rule in rules:
        matrix = np.atleast_2d(rule.A)
        lower = np.broadcast_to(rule.lb, (len(matrix),))
        upper = np.broadcast_to(rule.ub, (len(matrix),))
        for row, lo, hi in zip(matrix, lower, upper):
            if lo == hi:
                equality.append(row)
                eq_rhs.append(lo)
            else:
                if np.isfinite(hi):
                    inequality.append(row)
                    ineq_rhs.append(hi)
                if np.isfinite(lo):
                    inequality.append(-row)
                    ineq_rhs.append(-lo)
    return dict(A_eq=np.array(equality) if equality else None,
                b_eq=np.array(eq_rhs) if equality else None,
                A_ub=np.array(inequality) if inequality else None,
                b_ub=np.array(ineq_rhs) if inequality else None)


def _mv_violation(weights, rules, bounds):
    """Maximum absolute constraint violation, without modifying the weights."""
    w = np.asarray(weights)
    if w.shape != (len(bounds),) or not np.isfinite(w).all():
        return np.inf
    lower, upper = np.asarray(bounds).T
    error = max(0.0, float(np.max(lower-w)), float(np.max(w-upper)))
    for rule in rules:
        value = np.atleast_2d(rule.A) @ w
        error = max(error, float(np.max(rule.lb-value)), float(np.max(value-rule.ub)))
    return error


def _mv_feasible_start(n, rules, bounds):
    """Deterministic initialization independent of mu/cov and their units."""
    start = np.full(n, 1/n)
    if _mv_violation(start, rules, bounds) <= 1e-12:
        return start
    feasible = linprog(np.zeros(n), bounds=bounds, method="highs",
                       options={"primal_feasibility_tolerance": 1e-9,
                                "dual_feasibility_tolerance": 1e-9},
                       **_mv_linear_system(rules))
    if not feasible.success:
        if feasible.status == 2:
            raise ValueError(f"Mean-variance constraints are infeasible: {feasible.message}")
        raise RuntimeError(f"Mean-variance feasible initialization failed: {feasible.message}")
    if _mv_violation(feasible.x, rules, bounds) > 1e-9:
        raise RuntimeError("Mean-variance initialization did not produce a feasible portfolio.")
    return feasible.x


def mean_variance(mu, cov, risk_aversion, c):
    """Solve the original constrained MV problem with public risk aversion g.

    Objective: 0.5*w'cov*w - w'mu/g, multiplied by a positive numerical scale.
    Return raw, validated weights as a Series; never clip or renormalize a solve.
    Numerical feasibility is checked to 1e-9 in portfolio-weight units.
    """
    sigma, returns = _mv_coefficients(mu, cov, risk_aversion)
    tickers = list(mu.index)
    if any(t not in ASSETS for t in tickers):
        raise ValueError("Mean-variance received an unknown asset ticker.")
    for key in ("max_equity", "min_bonds", "max_other", "max_crypto", "max_weight_per_asset"):
        value = c.get(key)
        if (not isinstance(value, Real) or isinstance(value, (bool, np.bool_))
                or not np.isfinite(value)):
            raise ValueError(f"Mean-variance constraint {key} must be a finite number.")
    if c["max_weight_per_asset"] < 0:
        raise ValueError("Mean-variance per-asset upper bound makes the constraints infeasible.")
    n = len(tickers)
    rules = [LinearConstraint(np.ones(n), lb=1, ub=1)] + client_constraints(tickers, c)
    bounds = weight_bounds(tickers, c)
    start = _mv_feasible_start(n, rules, bounds)
    res = minimize(lambda w: 0.5*w @ sigma @ w - w @ returns, start,
                   jac=lambda w: sigma @ w - returns,
                   bounds=bounds, constraints=rules, method="SLSQP",
                   options={"ftol": 1e-12, "maxiter": 2000})
    if not res.success:
        raise RuntimeError(f"Mean-variance SLSQP failed (status {res.status}): {res.message}")
    violation = _mv_violation(res.x, rules, bounds)
    if violation > 1e-9:
        raise RuntimeError(f"Mean-variance returned invalid weights (maximum constraint violation {violation:.3g}).")
    if not np.isfinite(0.5*res.x @ sigma @ res.x - res.x @ returns):
        raise RuntimeError("Mean-variance returned a nonfinite objective value.")
    return pd.Series(res.x, index=tickers)


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

    Q = np.zeros((3 * n, 3 * n))
    Q[:n, :n] = cov.values                                    # only x has risk
    R = gamma * np.concatenate([mu.values, -costs, -costs])   # expected return minus costs
    objective = lambda X: 0.5 * X @ Q @ X - X @ R

    I = np.eye(n)
    rules = [
        LinearConstraint(np.concatenate([np.ones(n), costs, costs]), lb=1, ub=1),  # budget
        LinearConstraint(np.hstack([I, I, -I]), lb=x0, ub=x0),                    # link with x0
        LinearConstraint(np.concatenate([np.zeros(n), np.ones(2 * n)]), ub=max_turnover),  # turnover
    ] + client_constraints(tickers, c, extra_variables=2 * n)
    bounds = weight_bounds(tickers, c) + [(0, 1)] * (2 * n)  # 0 <= dx_minus, dx_plus <= 1
    start = np.concatenate([x0, np.zeros(2 * n)])
    res = minimize(objective, start, bounds=bounds, constraints=rules, method="SLSQP")
    return clean(res.x[:n], tickers)


def risk_contributions(weights, cov):
    """Share of the portfolio risk that comes from each asset (sums to 100%)."""
    w, sigma = np.asarray(weights), np.asarray(cov)
    return w * (sigma @ w) / (w @ sigma @ w)


def risk_parity(cov, c):
    """Equal Risk Contribution (ERC): every asset brings the same share of risk.
    With the client's constraints we get as close as possible to equal shares."""
    tickers = list(cov.index)
    n = len(tickers)
    sigma = cov.values
    objective = lambda w: 1000 * ((risk_contributions(w, sigma) - 1 / n) ** 2).sum()
    rules = [LinearConstraint(np.ones(n), lb=1, ub=1)] + client_constraints(tickers, c)
    res = minimize(objective, np.ones(n) / n, bounds=weight_bounds(tickers, c), constraints=rules,
                   method="SLSQP")
    if not res.success:
        raise RuntimeError(f"Risk parity SLSQP failed: {res.message}")
    return clean(res.x, tickers)


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


def frontier(mu, cov, c, gammas=np.geomspace(0.25, 200, 40)):
    """Efficient frontier and composition map (lab LN3): optimal portfolio for many gammas."""
    rows = []
    for g in gammas:
        w = mean_variance(mu, cov, g, c)
        ret, vol = portfolio_stats(w, mu, cov)
        rows.append({"Gamma": g, "Return": ret, "Volatility": vol, **w})
    return pd.DataFrame(rows)
