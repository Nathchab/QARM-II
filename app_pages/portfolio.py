"""Page 3: the optimal portfolio of the client, compared with other methods."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.app_helpers import (get_benchmark, get_choices, get_currency, get_data, get_profile,
                             get_valid_portfolios, get_views, loss_limit_note, method_label, next_button,
                             show_profile_messages)
from src.black_litterman import (CONFIDENCE_LEVELS, EXTREME_VIEW, MAX_VIEWS, market_weights, posterior_returns,
                                  prior_returns, restrict_views, view_check, view_problems)
from src.data import ASSETS, SHORT_TERM
from src.optimizer import beta, class_exposures, frontier, portfolio_stats, risk_contributions, tracking_error
from src.portfolios import METHODS, WINDOW, bad_year_loss, dominated_by, estimate, recommended_method

CLASS_COLORS = {"Equity": "#1f77b4", "Bond": "#2ca02c", "Other": "#ff7f0e", "Crypto": "#9467bd"}

profile = get_profile()
tickers, c = get_choices()
currency = get_currency()
benchmark_name, benchmark = get_benchmark(profile, tickers, c)

st.title("3. Your portfolio")
returns, rf, prices = get_data(currency)
mu, cov = estimate(returns[tickers])
rf_now = rf.iloc[-1] * 12
st.caption(f"Profile: {profile.label} (γ = {profile.gamma:.2f}) · Capacity: {profile.capacity_label} · "
           f"{len(tickers)} products · Currency: {currency} · Estimated on the last {WINDOW} months of data")
show_profile_messages(profile)

# Monthly returns of the last 60 months, to compare with the benchmark (beta, tracking error)
past = returns[tickers].dropna().iloc[-WINDOW:]
benchmark_returns = returns.loc[past.index, benchmark.index].fillna(0) @ benchmark

# ---------- Only for experienced investors who want it (QH4): their views, with Black-Litterman (LN4) ----------
prior = prior_returns(cov, rf_now)
if profile.wants_views:
    st.subheader("Your views on the markets (Black-Litterman)")
    st.write("You told us that you have your own views (QH4). Black-Litterman (lecture LN4) moves the market "
             "equilibrium returns towards your views, as much as your confidence allows. Add a view only where you "
             "have a real conviction. **Absolute**: “GLD will return 6% per year”. **Relative**: “EEM will beat SPY "
             "by 3% per year”. With at least one valid view, Black-Litterman becomes our recommendation.")
    saved_views = list(restrict_views(st.session_state.get("views", ()), mu.index))
    products = list(mu.index)
    levels = list(CONFIDENCE_LEVELS)
    n = st.number_input("Number of views", 0, MAX_VIEWS, len(saved_views), key="n_views")
    views = []
    for k in range(int(n)):
        asset, other, q, conf = saved_views[k] if k < len(saved_views) else (
            products[0], None, round(float(prior.iloc[0]) * 100, 1) / 100, CONFIDENCE_LEVELS["Medium"])
        st.markdown(f"**View {k + 1}**")
        col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 3])
        kind = col1.selectbox("Type", ["Absolute", "Relative"], index=0 if other is None else 1, key=f"v{k}_type")
        asset = col2.selectbox("Product", products, index=products.index(asset), key=f"v{k}_asset")
        if kind == "Relative":
            others = [t for t in products if t != asset]
            other = other if other in others else others[0]
            other = col3.selectbox("will beat", others, index=others.index(other), key=f"v{k}_other")
            label = "by (% per year)"
        else:
            other = None
            col3.write("")
            label = "will return (% per year)"
        q = col4.number_input(label, -30.0, 50.0, float(round(q * 100, 1)), 0.5, key=f"v{k}_q") / 100
        level = col5.select_slider("Confidence (QH6)", levels,
                                   value=min(levels, key=lambda name: abs(CONFIDENCE_LEVELS[name] - conf)),
                                   key=f"v{k}_conf")
        view = (asset, other, q, CONFIDENCE_LEVELS[level])
        check = view_check(cov, prior, view)
        what = f"π_{asset}" if other is None else f"π_{asset} − π_{other}"
        st.caption(f"Equilibrium: {what} = {check['equilibrium']:.1%} · your view is "
                   f"{q - check['equilibrium']:+.1%} away ({check['gap']:+.1f} standard deviations of the view, "
                   f"volatility {check['volatility']:.1%}).")
        if check["extreme"]:
            st.warning(f"View {k + 1} is more than {EXTREME_VIEW:g} standard deviations away from the market "
                       "equilibrium: are you sure? With a high confidence it will dominate the portfolio.")
        views.append(view)
    problems = view_problems(views)
    for problem in problems:
        st.error(problem)
    if problems:
        st.info("Your views are not used until these problems are fixed.")
    st.session_state["views"] = tuple(views) if not problems else ()
    st.caption("Each view is a row of P (+1 on the product, −1 on the second product of a relative view) and "
               "a value of Q. Its uncertainty is Ω = τ·(PΣPᵀ)·(1 − c)/c, with c = 20%, 50%, 80% or 95%: "
               "Medium gives the view the same uncertainty as the equilibrium.")

views = restrict_views(get_views(), mu.index)
recommended = recommended_method(views)
portfolios, gamma = get_valid_portfolios(profile, tickers, c)
methods = [m for m in METHODS if m in portfolios]
saved = st.session_state.get("method", recommended)
default = saved if saved in methods else recommended if recommended in methods else methods[0]
method = st.radio("Choose a method", methods, horizontal=True, index=methods.index(default),
                  format_func=lambda m: method_label(m) + (" – recommended" if m == recommended else ""))
st.session_state["method"] = method
weights = portfolios[method]

# ---------- Our recommendation: mean-variance with the market equilibrium returns ----------
if method in ("Equilibrium", "Black-Litterman"):
    st.write("**Where do the expected returns come from?** Not from the last 5 years: their average is a very noisy "
             "guess of the future (lab LN3). We use the **market equilibrium** π = r_f + δΣw_mkt, the returns that "
             "make the market portfolio optimal, i.e. what the market as a whole expects for the risk of each "
             "product. Then the mean-variance optimiser (Markowitz) chooses the best mix for your γ and your limits."
             + (" Black-Litterman moves π towards your views." if method == "Black-Litterman" else ""))
    with st.expander("See the equilibrium returns"):
        st.dataframe(pd.DataFrame({
            "Product": [ASSETS[t][0] for t in mu.index],
            "Market weight": market_weights(mu.index),
            "Equilibrium return": prior,
            f"Historical return ({WINDOW} months)": mu,
        }).style.format({"Market weight": "{:.0%}", "Equilibrium return": "{:.1%}",
                         f"Historical return ({WINDOW} months)": "{:.1%}"}))

# ONE forecast of the expected returns for every method, so that the portfolios can be compared:
# the market equilibrium (plus the client's views). The average of the last 60 months is NOT a forecast.
forecast = posterior_returns(cov, rf_now, views)

# ---------- Key numbers ----------
ret, vol = portfolio_stats(weights, forecast, cov)
exposure = class_exposures(weights)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Expected return", f"{ret:.1%}", help="With our forecast (market equilibrium + your views), "
            "before our fee. The same forecast is used for every method.")
col2.metric("Volatility", f"{vol:.1%}")
col3.metric("Sharpe ratio", f"{(ret - rf_now) / vol:.2f}" if vol > 0 else "–")
col4.metric("Beta vs benchmark", f"{beta(past @ weights, benchmark_returns):.2f}",
            help=f"Benchmark: {benchmark_name} (you can change it on the backtest page)")
loss = bad_year_loss(weights, forecast, cov)
rec_loss = bad_year_loss(portfolios.get(recommended, weights), forecast, cov)
note = loss_limit_note(profile, gamma, rec_loss, currency)
if profile.max_loss is not None:
    if loss >= profile.max_loss - 1e-4:
        st.caption(f"In a bad year (5% chance) this portfolio could lose about {-loss:.1%}: within the "
                   f"{-profile.max_loss:.0%} you can tolerate (QE2).")
    elif method != recommended and loss < rec_loss - 1e-4:
        st.warning(f"In a bad year (5% chance) this portfolio could lose about {-loss:.0%}, more than the "
                   f"{-profile.max_loss:.0%} you can tolerate (QE2). Choose the recommended method instead.")
if note:
    (st.warning if rec_loss < profile.max_loss - 1e-4 else st.info)(note)
col1, col2, col3, col4 = st.columns(4)
col1.metric("In equities", f"{exposure['Equity']:.0%}", help=f"Your maximum: {c['max_equity']:.0%}")
col2.metric("In bonds", f"{exposure['Bond']:.0%}", help=f"Your minimum: {c['min_bonds']:.0%}")
col3.metric("Largest product", f"{weights.max():.0%}")
col4.metric("Effective number of products", f"{1 / (weights @ weights):.1f}", help="1 / Σ wᵢ²")
# A limit is binding when the portfolio sits exactly on it (a limit of 0% is not worth mentioning)
LIMITS = {"max_equity": ("Equity", "maximum in equities"), "min_bonds": ("Bond", "minimum in bonds"),
          "max_other": ("Other", "maximum in gold and commodities"), "max_crypto": ("Crypto", "maximum in crypto")}
binding = [name for key, (kind, name) in LIMITS.items()
           if abs(exposure[kind] - c[key]) <= 1e-6 and not (key in ("min_bonds", "max_crypto") and c[key] == 0)]
if c.get("min_short_term", 0) > 0 and abs(weights.get(SHORT_TERM, 0) - c["min_short_term"]) <= 1e-6:
    binding.append("minimum in short-term Treasuries")
if binding and method != "Risk parity (ERC)":
    st.caption("Your limits are binding: " + ", ".join(binding) + ". Without them, the "
               "optimiser would choose a different mix for your γ.")

# ---------- Weights and risk contributions ----------
table = pd.DataFrame({
    "Product": [ASSETS[t][0] for t in weights.index],
    "Class": [ASSETS[t][1] for t in weights.index],
    "Weight": weights,
    "Risk contribution": risk_contributions(weights, cov),
})
col1, col2 = st.columns(2)
col1.plotly_chart(px.bar(table, x=table.index, y="Weight", color="Class", title="Weights",
                         color_discrete_map=CLASS_COLORS,
                         text_auto=".0%").update_layout(yaxis_tickformat=".0%", xaxis_title=None))
col2.plotly_chart(px.bar(table, x=table.index, y="Risk contribution", color="Class",
                         color_discrete_map=CLASS_COLORS,
                         title="Where does the risk come from?",
                         text_auto=".0%").update_layout(yaxis_tickformat=".0%", xaxis_title=None))

# ---------- Is the recommended portfolio efficient? ----------
better = dominated_by(portfolios, forecast, cov, recommended) if recommended in portfolios else []
if better:
    st.warning(f"With our forecast, {', '.join(method_label(m) for m in better)} gives a higher expected return "
               f"with less risk than {method_label(recommended)}: choose it instead.")
else:
    st.success(f"{method_label(recommended)} is efficient: with our forecast, no other method gives a higher "
               "expected return with less risk (it is the optimal portfolio for your γ and your limits).")

# ---------- Efficient frontier: all methods with the same forecast ----------
try:
    front = frontier(forecast, cov, c)
except (ValueError, RuntimeError, np.linalg.LinAlgError):
    front = pd.DataFrame()
    st.caption("The efficient frontier could not be calculated with these inputs.")
fig = go.Figure()
if not front.empty:
    fig.add_scatter(x=front["Volatility"], y=front["Return"], mode="lines", name="Efficient frontier (your limits)")
fig.add_scatter(x=np.sqrt(np.diag(cov)), y=forecast, mode="markers+text", text=list(mu.index),
                textposition="top center", name="Single products", marker=dict(color="lightgrey"))
for m, w in portfolios.items():
    r, v = portfolio_stats(w, forecast, cov)
    fig.add_scatter(x=[v], y=[r], mode="markers", name=method_label(m),
                    marker=dict(size=18 if m == method else 11, symbol="star" if m == method else "circle"))
fig.update_layout(title="Efficient frontier with our forecast (the star is the method you selected)",
                  xaxis_title="Volatility", yaxis_title="Expected return", xaxis_tickformat=".0%", yaxis_tickformat=".1%")
st.plotly_chart(fig)
st.caption("Mean-variance and resampled mean-variance trust the average returns of the last 5 years, so with our "
           "forecast they are below the frontier. Risk parity uses no expected returns at all: it gives up some "
           "expected return to spread the risk.")

# ---------- Compare the methods ----------
st.subheader("Compare the methods")
compare = pd.DataFrame(portfolios)
compare.index = [f"{t} ({ASSETS[t][0]})" for t in compare.index]
stats = pd.DataFrame({m: {"Expected return (our forecast)": portfolio_stats(w, forecast, cov)[0],
                          "Volatility": portfolio_stats(w, forecast, cov)[1],
                          f"Average return of the last {WINDOW} months (past)": portfolio_stats(w, mu, cov)[0],
                          "Beta vs benchmark": beta(past @ w, benchmark_returns),
                          "Tracking error vs benchmark": tracking_error(past @ w, benchmark_returns)}
                      for m, w in portfolios.items()})
table = pd.concat([compare, stats])
table.columns = [method_label(m) for m in table.columns]
st.dataframe(table.style.format("{:.1%}").format("{:.2f}", subset=pd.IndexSlice[["Beta vs benchmark"], :]))
st.caption(f"Benchmark: {benchmark_name}. Beta and tracking error (lab LN2) on the last {WINDOW} months. "
           f"The past average is not a forecast: mean-variance was built to be the best portfolio for exactly these "
           "past returns, so of course it looks best on them. The backtest (page 5) compares the methods fairly, "
           "out of sample.")

# ---------- Composition map (lab LN3): sensitivity to gamma ----------
if not front.empty:
    st.subheader("What if your γ was different?")
    st.write("Composition map (as in lab LN3): the optimal weights for every γ, with your limits and our forecast "
             "(this is the recommended method). The dashed line is your γ.")
    area = px.area(front, x="Gamma", y=list(mu.index), log_x=True)
    area.add_vline(x=min(gamma, front["Gamma"].max()), line_dash="dash")
    area.update_layout(yaxis_title="Weight", yaxis_tickformat=".0%", xaxis_title="γ (risk aversion)")
    st.plotly_chart(area)

next_button("app_pages/plan.py", "Next: your plan (projection, goal, orders, report)")
