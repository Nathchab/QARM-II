"""Page 3: the optimal portfolio of the client, compared with other methods."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.app_helpers import (explain, get_benchmark, get_choices, get_currency, get_data, get_profile,
                             get_valid_portfolios, loss_limit_note, method_label, next_button,
                             show_profile_messages)
from src.black_litterman import (CONFIDENCE_LEVELS, DELTA, EXTREME_VIEW, MARKET_PORTFOLIO, MAX_VIEWS,
                                  posterior_returns, prior_returns, restrict_views, view_check, view_problems)
from src.data import ASSETS
from src.optimizer import (beta, class_exposures, frontier, home_row, portfolio_stats, region_exposures,
                           risk_contributions, short_term_row, tracking_error)
from src.planning import MANAGEMENT_FEE
from src.portfolios import METHODS, WHY, WINDOW, bad_year_loss, dominated_by, estimate, recommended_method

CLASS_COLORS = {"Equity": "#1f77b4", "Bond": "#2ca02c", "Other": "#ff7f0e", "Crypto": "#9467bd"}

profile = get_profile()
tickers, c = get_choices()
currency = get_currency()
benchmark_name, benchmark = get_benchmark(profile, tickers, c)

st.title("3. Your portfolio")
returns, rf, prices = get_data(currency)
mu, cov, market_cov = estimate(returns, tickers)
rf_now = rf.iloc[-1] * 12
st.caption(f"Profile: {profile.label} (γ = {profile.gamma:.2f}) · Capacity: {profile.capacity_label} · "
           f"{len(tickers)} products · Currency: {currency} · Estimated on the last {WINDOW} months of data")
show_profile_messages(profile)

# Monthly returns of the last 60 months, to compare with the benchmark (beta, tracking error)
past = returns[tickers].dropna().iloc[-WINDOW:]
benchmark_returns = returns.loc[past.index, benchmark.index].fillna(0) @ benchmark

# ---------- Only for experienced investors who want it (QH4): their views, with Black-Litterman ----------
prior = prior_returns(market_cov, rf_now)  # market equilibrium returns
if profile.wants_views:
    st.subheader("Your views on the markets (Black-Litterman)")
    st.write("You told us that you have your own views (QH4). Black-Litterman moves the market "
             "equilibrium returns towards your views, as much as your confidence allows. Add a view only where you "
             "have a real conviction. **Absolute**: “GLD will return 6% per year”. **Relative**: “EEM will beat SPY "
             "by 3% per year”. With at least one valid view, Black-Litterman becomes our recommendation"
             + (" (not for you: you may need this money within a year)." if profile.preserve_capital else "."))
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
        st.caption(f"Equilibrium: {what} = {check['equilibrium']:.1%} (known to about ±{check['uncertainty']:.1%}) · "
                   f"your view is {q - check['equilibrium']:+.1%} away, {abs(check['gap']):.1f} times this uncertainty.")
        if check["extreme"]:
            st.warning(f"View {k + 1} is more than {EXTREME_VIEW:g} times the uncertainty away from the market "
                       "equilibrium: are you sure? With a high confidence it will dominate the portfolio.")
        views.append(view)
    problems = view_problems(views)
    for problem in problems:
        st.error(problem)
    if problems:
        st.info("Your views are not used until these problems are fixed.")
    st.session_state["views"] = tuple(views) if not problems else ()
    if views and not problems:  # what the views do to the expected returns, before any optimisation
        after = posterior_returns(prior, cov, tuple(views))
        st.dataframe(pd.DataFrame({"Market equilibrium": prior, "With your views": after, "Change": after - prior})
                     .T.style.format("{:+.1%}", subset=pd.IndexSlice[["Change"], :])
                     .format("{:.1%}", subset=pd.IndexSlice[["Market equilibrium", "With your views"], :]))
        st.caption("Expected returns before and after your views. A view on one product also moves the products "
                   "that are correlated with it.")
    st.caption("Each view is a row of P (+1 on the product, −1 on the second product of a relative view) and "
               "a value of Q. Its uncertainty is Ω = τ·(PΣPᵀ)·(1 − c)/c, with c = 20%, 50%, 80% or 95%: "
               "Medium gives the view the same uncertainty as the equilibrium.")

# ONE forecast of the expected returns for every method, so that the portfolios can be compared:
# the market equilibrium (plus the client's views). The average of the last 60 months is NOT a forecast.
portfolios, gamma, forecast, cov = get_valid_portfolios(profile, tickers, c)
recommended = recommended_method(profile, portfolios, forecast, cov)
methods = [m for m in METHODS if m in portfolios]
saved = st.session_state.get("method") or recommended  # the client's own choice, otherwise our recommendation
default = saved if saved in methods else recommended if recommended in methods else methods[0]
# ---------- Our recommendation, clearly visible ----------
with st.container(border=True):
    st.markdown(f"### ⭐ Our recommendation for you: :orange[{method_label(recommended)}]")
    st.write(WHY[recommended])
    st.caption("You can look at the other methods below: they are shown to compare.")
method = st.radio("Choose a method to look at", methods, horizontal=True, index=methods.index(default),
                  format_func=lambda m: f"⭐ :orange[**{method_label(m)}** (recommended)]" if m == recommended
                  else method_label(m))
if method != recommended:
    st.warning(f"You are looking at {method_label(method)}, not at our recommendation.")
# We remember the method only if the client chose another one: otherwise he follows our recommendation,
# even when it changes (for example after he enters views)
st.session_state["method"] = method if method != recommended else None
weights = portfolios[method]

# ---------- Our recommendation: mean-variance with the market equilibrium returns ----------
if method == "Minimum variance":
    st.write("**Minimum variance:** the portfolio with the **lowest volatility** inside your limits. It does not use "
             "any forecast of returns, so it cannot be wrong because of a bad forecast. "
             "It is what we recommend when the money may be needed within a year.")
if method in ("Equilibrium", "Black-Litterman"):
    st.write("**Where do the expected returns come from?** Not from the last 5 years: their average is a very noisy "
             "guess of the future. We use the **market equilibrium** πᵢ = r_f + δ·cov(rᵢ, r_market), the returns "
             "that make the market portfolio optimal: a product is expected to earn more only if it moves with the "
             "market as a whole. Then the mean-variance optimiser (Markowitz) chooses the best mix for your γ and your limits."
             + (" Black-Litterman moves π towards your views." if method == "Black-Litterman" else ""))
    with st.expander("See the equilibrium returns"):
        st.dataframe(pd.DataFrame({
            "Product": [ASSETS[t][0] for t in mu.index],
            "Covariance with the market": market_cov,
            "Equilibrium return": prior,
            f"Historical return ({WINDOW} months)": mu,
        }).style.format({"Covariance with the market": "{:.4f}", "Equilibrium return": "{:.1%}",
                         f"Historical return ({WINDOW} months)": "{:.1%}"}))
        st.caption("Market portfolio: " + ", ".join(f"{w:.0%} {t}" for t, w in MARKET_PORTFOLIO.items())
                   + f" (our own round numbers). δ = {DELTA}.")

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
explain(profile, "**Expected return**: what the portfolio should earn per year on average (our forecast, not a "
                 "promise). **Volatility**: how much it moves up and down in a typical year; about 2 years out of 3 "
                 "the return stays within ± one volatility of the expected return. **Sharpe ratio**: extra return "
                 "per unit of risk, compared with a riskless investment (higher is better). **Beta**: if the "
                 "benchmark moves by 1%, your portfolio moves on average by beta %.")
if ret - MANAGEMENT_FEE < rf_now:  # honest advice: this portfolio is not worth our fee
    st.warning(f"After our fee of {MANAGEMENT_FEE:.2%} per year, this portfolio is expected to earn less than a "
               f"riskless investment ({rf_now:.1%} per year today). For this money, a savings account or a "
               "money-market fund is a better choice than a managed portfolio.")
loss = bad_year_loss(weights, forecast, cov)
rec_loss = bad_year_loss(portfolios.get(recommended, weights), forecast, cov)
note = loss_limit_note(profile, gamma, rec_loss, currency, recommended)
if profile.max_loss is not None:
    if loss >= 0:
        st.caption("Even in a bad year (5% chance) this portfolio should not lose money: within the "
                   f"{-profile.max_loss:.0%} {profile.loss_reason}.")
    elif loss >= profile.max_loss - 1e-4:
        st.caption(f"In a bad year (5% chance) this portfolio could lose about {-loss:.1%}: within the "
                   f"{-profile.max_loss:.0%} {profile.loss_reason}.")
    elif method != recommended and loss < rec_loss - 1e-4:
        st.warning(f"In a bad year (5% chance) this portfolio could lose about {-loss:.0%}, more than the "
                   f"{-profile.max_loss:.0%} {profile.loss_reason}. Choose the recommended method instead.")
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
if c.get("min_short_term", 0) > 0 and abs(short_term_row(tickers) @ weights - c["min_short_term"]) <= 1e-6:
    binding.append("minimum in short-term government bonds")
if c.get("min_home_equity", 0) > 0 and abs(home_row(tickers, c) @ weights) <= 1e-6:
    binding.append(f"minimum share of your equities in {c['region']}")
if binding and method not in ("Risk parity (ERC)", "Minimum variance"):
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
                         color_discrete_map=CLASS_COLORS, text_auto=".0%").update_layout(
    yaxis_tickformat=".0%", xaxis_title=None).update_traces(textposition="outside", cliponaxis=False))
col2.plotly_chart(px.bar(table, x=table.index, y="Risk contribution", color="Class",
                         color_discrete_map=CLASS_COLORS,
                         title="Where does the risk come from?", text_auto=".0%").update_layout(
    yaxis_tickformat=".0%", xaxis_title=None).update_traces(textposition="outside", cliponaxis=False))
explain(profile, "**Weights** (left): how much of your money goes into each product. **Risk contributions** "
                 "(right): how much of the ups and downs of the portfolio comes from each product. Equities often "
                 "bring most of the risk even with a smaller weight, because they move much more than bonds.")

# ---------- Regions: the equities alone (what QF3 is about) and the whole portfolio ----------
equities = weights[[t for t in weights.index if ASSETS[t][1] == "Equity"]]
parts = {"Whole portfolio": region_exposures(weights)}
if equities.sum() > 0.005:
    parts["Your equities only"] = region_exposures(equities / equities.sum())
regions = pd.DataFrame(parts)
regions = regions[regions.max(axis=1) > 0.005]
col1, col2 = st.columns(2)
classes = {k: v for k, v in exposure.items() if v > 0.0005}
col1.plotly_chart(px.pie(names=list(classes), values=list(classes.values()), hole=0.5, title="By asset class",
                         color=list(classes), color_discrete_map=CLASS_COLORS).update_traces(textinfo="percent+label"))
col2.plotly_chart(px.bar(regions, orientation="h", barmode="group", text_auto=".0%",
                         title="Where is your money invested?").update_layout(
    xaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None, yaxis=dict(autorange="reversed"), legend_title=None))
explain(profile, "The left chart splits your money between equities, bonds, gold/commodities and crypto. The right "
                 "chart looks inside the funds (approximate, rounded index weights), for your equities alone and for "
                 "the whole portfolio. Two things to know: the United States are about two thirds of the world stock "
                 "market, so even a worldwide equity portfolio is mostly North American; and your bonds and cash are "
                 f"in {currency}, the safe part of the portfolio, so the whole portfolio leans towards the home of "
                 "this currency. Gold, commodities and crypto belong to no region."
                 + (f" You asked for {c['region']}: at least {c['min_home_equity']:.0%} of your equities are there."
                    if c.get("min_home_equity", 0) > 0 else ""))

# ---------- Is the recommended portfolio efficient? ----------
better = dominated_by(portfolios, forecast, cov, recommended) if recommended in portfolios else []
if recommended == "Risk parity (ERC)":
    st.info("We recommend risk parity because your risk answers contradict each other, so we cannot measure your γ "
            "reliably: risk parity needs neither γ nor a forecast of returns, it simply spreads the risk evenly. "
            + (f"With our forecast, {', '.join(method_label(m) for m in better)} would look better, but only if "
               "the forecast and your γ are right." if better else ""))
elif better:
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
    fig.add_scatter(x=[v], y=[r], mode="markers", name=method_label(m) + (" ⭐ recommended" if m == recommended else ""),
                    marker=dict(size=20 if m == recommended else 11, symbol="star" if m == recommended else "circle",
                                color="#e4572e" if m == recommended else None,
                                line=dict(width=3, color="black") if m == method else None))
fig.update_layout(title="Efficient frontier with our forecast (orange star: our recommendation)",
                  xaxis_title="Volatility", yaxis_title="Expected return", xaxis_tickformat=".0%", yaxis_tickformat=".1%")
st.plotly_chart(fig)
st.caption("Minimum variance is the left end of the frontier. Mean-variance and resampled mean-variance trust the "
           "average returns of the last 5 years, so with our forecast they are below the frontier. Risk parity uses "
           "no expected returns at all: it gives up some expected return to spread the risk.")
explain(profile, "Each point is a portfolio: to the right = more risk (volatility), higher = more expected return. "
                 "The **efficient frontier** (blue line) is the best you can get: for each level of risk, the "
                 "highest expected return possible with your products and limits. A portfolio below the line is "
                 "not efficient: another one gives more return for the same risk. The grey points are the single "
                 "products: combining them gives more return for less risk (diversification).")

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
rec_col = method_label(recommended)
st.dataframe(table.style.format("{:.1%}").format("{:.2f}", subset=pd.IndexSlice[["Beta vs benchmark"], :])
             .set_properties(subset=[rec_col], **{"background-color": "#fde4dc", "font-weight": "bold"}))
st.caption(f"Benchmark: {benchmark_name}. Beta and tracking error on the last {WINDOW} months. "
           f"The past average is not a forecast: mean-variance was built to be the best portfolio for exactly these "
           "past returns, so of course it looks best on them. The backtest (page 5) compares the methods fairly, "
           "out of sample.")
explain(profile, "Each column is one method (ours is highlighted): the weights of each product, then its numbers. "
                 "**Tracking error**: how far the portfolio can drift away from the benchmark in a year.")

# ---------- Composition map (lab LN3): sensitivity to gamma ----------
if not front.empty:
    st.subheader("What if your γ was different?")
    st.write("Composition map: the optimal weights for every γ, with your limits and our forecast "
             "(the market equilibrium method; with a very high γ it becomes the minimum variance portfolio). "
             "The dashed line is your γ.")
    area = px.area(front, x="Gamma", y=list(mu.index), log_x=True)
    area.add_vline(x=min(gamma, front["Gamma"].max()), line_dash="dash")
    area.update_layout(yaxis_title="Weight", yaxis_tickformat=".0%", xaxis_title="γ (risk aversion)")
    st.plotly_chart(area)
    explain(profile, "From left to right the investor is more and more prudent (higher γ): the colours show how the "
                     "optimal portfolio slowly moves from equities to bonds and short-term bonds. It shows that your "
                     "portfolio changes smoothly with your answers, without jumps.")

next_button("app_pages/plan.py", "Next: your plan (projection, goal, orders, report)")
