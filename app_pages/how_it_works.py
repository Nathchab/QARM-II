"""Page 7: short explanation of the methods, data and limitations (useful for questions)."""
import streamlit as st

from src.black_litterman import CONFIDENCE_LEVELS, DELTA, MARKET_PORTFOLIO, TAU
from src.client_profile import GAMMA_MAX, GAMMA_MIN
from src.data import ASSETS

st.title("How it works")

st.header("1. Your profile")
st.write("The questionnaire (QARM II questionnaire, sections B to F and H) measures two different things, "
         "and each one has its own job in the portfolio.")
st.markdown("""
- **Risk tolerance → γ.** RT is the average of the four answers of part 1 (QE1–QE4, each scored 0–100).
  It is converted into the risk aversion of the mean-variance optimiser:
""")
st.latex(rf"\gamma(RT) = {GAMMA_MAX:g}\times\left(\tfrac{{{GAMMA_MIN:g}}}{{{GAMMA_MAX:g}}}\right)^{{RT/100}}")
st.markdown("""
  The anchors are calibrated on QE3: with the utility U = μ − γ/2·σ², the client is indifferent between
  two options when γ = 2(μ₂ − μ₁)/(σ₂² − σ₁²). The four QE3 profiles give the thresholds 8, 2.67 and 1.6,
  and γ evaluated at the RT score of each answer falls inside the interval this answer reveals.
- **Risk capacity → allocation limits.** Age, income stability, emergency savings and commitments give a
  score from 0 to 9 (Low 0–2, Limited 3–4, Moderate 5–6, High 7–9). Low savings or heavy commitments cap the
  capacity at Limited. Each tier gives a maximum in equities, a minimum in bonds, maximums in gold/commodities
  and crypto, and the largest loss the client can afford in a bad year (Low 10%, Limited 15%, Moderate 20%,
  High: no limit). The horizon (QD1) and the possible withdrawals (QD2) give their own equity maximum and bond minimum.
  **The most restrictive source applies.** The client can tighten these limits, never relax them.
- **Money needed soon goes into short-term government bonds** (SHY in USD, Swiss government bonds 0-3 years
  CSBGC3.SW in CHF). A mean-variance optimiser looks at one period
  only: for the bond minimum it would happily choose 20-year Treasuries (TLT), which can lose a lot when rates
  rise (about 30% in 2022). So money that may be needed within 3 years (QD1) or withdrawn within 12 months (QD2)
  must be in short-term bonds: at least 60%/40% for QD1 "within 1 year"/"1–3 years" and 50%/25%/10% for QD2
  ">50%"/"25–50%"/"10–25%".
- **Your loss limit is respected.** Two answers give a largest loss: the loss the client says he can tolerate
  (QE2) and the loss he can afford (his financial capacity). The stricter of the two is a hard limit: in a bad
  year (5% chance, normal distribution) the recommended portfolio should not lose more. So its volatility must be
  at most |limit| / 1.645 (5% → 3.0%, 10% → 6.1%, 15% → 9.1%, 20% → 12.2%, 30% → 18.2%). If the portfolio of γ
  is riskier, we raise γ to the smallest value that respects the limit ("targeting a volatility", solved by
  bisection). The capacity limit is needed because a minimum in bonds does not limit the risk by itself: without
  it, a bold client with a low capacity got 40% in 20-year bonds and a volatility of 12.6%, almost as much as a
  client with a high capacity. If even the safest portfolio inside the limits is too risky, we recommend that
  safest portfolio and say so. Over the past 15 years, the 12-month loss of the most prudent portfolios was
  larger than their 5% limit about 5% of the time (2022, when equities and bonds fell together): it is a limit
  with a 95% confidence, not a guarantee.
- **Region (QF3) and currency.** Equities follow the region the client prefers (Switzerland: SMI, Swiss mid caps
  + world fund ACWI; Europe: VGK + ACWI; USA: S&P 500, small caps + EFA; World: S&P 500 + EFA), and **at least
  50% of the equities** must be in that region, looking through the funds (ACWI is about 2% Swiss). The other half
  stays worldwide: the SMI alone is 20 companies, and Nestlé, Novartis and Roche are about 40% of it.
  Bonds and cash are the safe part of the portfolio, so they follow its **currency**: in CHF we use Swiss
  government and corporate bonds (traded in CHF on SIX), so the safe part has no currency risk. A portfolio in
  USD keeps US bonds and also gets a government bond fund of the client's region: Swiss government bonds
  (Switzerland) or international bonds hedged to USD, about 60% European (Europe).
- **Diversification rules.** At most 40% in one product, except the short-term government bonds (almost cash:
  no maximum). Narrow products (one sector or country: QQQ, IWM, EWL, EWU, EWG, EWJ, EEM, VNQ, HYG, EMB, SLV,
  Swiss mid caps) at most 15% each (core-satellite).
  Without this rule the most tolerant clients got 35% in real estate and 21% in emerging markets.
- **Why capacity does not set γ.** γ comes from the tolerance answers only. If capacity also gave a γ floor for
  everybody, most clients' tolerance would no longer change anything. Capacity only draws the box (allocation
  limits and largest loss): a tolerant client with a low capacity gets the most dynamic portfolio *inside* it.
- **Warnings** are kept for material facts only: emergency savings below 3 months, significant commitments,
  money needed within a year, or likely withdrawals above 25%. Inconsistent tolerance answers are only
  flagged when the final γ is at least two QE3 intervals away from the QE3 answer.
- **Which method we recommend.** Each method has its own type of client:

  | Client | Recommended method | Why |
  |---|---|---|
  | Needs the money within a year (QD1), or may withdraw more than half within 12 months (QD2) | Minimum variance | capital preservation: over such a short time only the risk matters; needs no return forecast |
  | Experienced investor who entered views (QH4) | Black-Litterman | uses his views, weighted by his confidence |
  | Risk answers that contradict each other (γ two QE3 intervals away from QE3) | Risk parity (ERC) | needs neither γ nor a return forecast; only if it respects QE2 |
  | Everybody else | Market equilibrium | mean-variance with the equilibrium returns π, the client's γ and limits |

  Mean-variance and resampled mean-variance with past returns are shown only to compare (they show why we do not
  trust past averages).
- **One forecast for all methods.** Every expected return shown in the app (portfolio page, frontier, plan) uses
  the same forecast: the market equilibrium (plus the views of an experienced client, if any). With this forecast the recommended portfolio
  is the optimal one for γ and the limits, so it is efficient: no other method has a higher expected return with
  a lower risk (the portfolio page checks it). The average of the last 60 months is shown only as information.
- **Products.** The asset classes chosen in QF1 give the recommended products; crypto (QF2) is also capped by
  capacity. Every client also gets the short-term government bonds, even if he did not choose them: they are the
  "cash" of the portfolio, and without them the optimiser cannot lower the risk to the level of a prudent
  client (a client with only equity funds would hold at least 60% in equities whatever his γ).
- **Risk parity and minimum variance** do not use γ: only the covariance and the allocation limits.
- **Honest about our fee.** If the expected return of the portfolio after our fee is below the riskless rate, the
  portfolio page tells the client that a savings account is a better choice (typically money needed within a year).
""")

st.header("2. Data")
st.write("The app downloads the daily prices directly from Yahoo Finance when it starts (again every day, or "
         "with the button 'Update data now'). Nothing has to be run before. We use monthly returns; "
         "expected returns and covariances are estimated on the last 60 months. Products ending in .SW trade in "
         "CHF on the Swiss exchange; the others in USD. Prices are converted with the USD/CHF rate when needed.")
st.table({"Ticker": list(ASSETS), "Product": [a[0] for a in ASSETS.values()],
          "Class": [a[1] for a in ASSETS.values()]})

st.header("3. Methods")
st.subheader("Mean-variance (Markowitz 1952)")
st.latex(r"\min_x \; \tfrac{1}{2} x^\top \Sigma x - \tfrac{1}{\gamma} x^\top \mu "
         r"\quad \text{s.t. } \textstyle\sum x_i = 1,\; 0 \le x_i \le x^+,\; \text{class constraints}")
st.write("γ is the risk aversion of the client (from the questionnaire). We use this problem with "
         "two different forecasts μ:")
st.write("- **Market equilibrium (our recommendation):** μ = π with πᵢ = r_f + δ·cov(rᵢ, r_market), the returns "
         f"that make the market portfolio optimal for the average investor (\"reverse optimisation\", δ = {DELTA}). "
         "For the market portfolio itself this is π = r_f + δΣw_market; a product earns more only if it moves with "
         "the market (CAPM). The market portfolio is the same for every client ("
         + ", ".join(f"{w:.0%} {t}" for t, w in MARKET_PORTFOLIO.items()) + "), so the expected return of a product "
         "does not depend on the other products he chose, and the same answers give about the same risk in USD and "
         "in CHF. These returns are far more stable than past averages, so the portfolio does not jump from one "
         "month to the next.\n"
         "- **Past returns:** μ = average of the last 60 months, the textbook version. Shown for comparison: it is "
         "very sensitive to estimation errors.")

st.subheader("Resampled mean-variance (Michaud 2007)")
st.write("Expected returns are very uncertain, so mean-variance weights are unstable. We simulate 100 new "
         "samples of 60 months from N(μ, Σ), optimise each one and average the weights.")

st.subheader("Black-Litterman (1992): only for experienced investors with views")
st.latex(r"\pi_i = r_f + \delta\,\mathrm{cov}(r_i, r_{market})")
st.latex(r"\mu_{BL} = \pi + \Gamma P^\top \left(P \Gamma P^\top + \Omega\right)^{-1} (Q - P\pi),"
         r"\qquad \Gamma = \tau\Sigma")
st.write(f"π: equilibrium returns of the market portfolio (δ = {DELTA}, τ = {TAU}). Black-Litterman starts from "
         "π and moves it towards the client's views. Without views μ_BL = π, so for the other clients we simply "
         "use π (the market equilibrium method above) and do not show Black-Litterman at all.")
st.write("**Views (experienced investors only).** A client is experienced if he has invested for at least 2 years, "
         "rates his knowledge at least intermediate (QB1, QC1) and answers the diversification question QC3 "
         "correctly. If he also says he has convictions (QH4), he can enter up to 5 views on the portfolio page. "
         "An absolute view (“GLD will return 6%”) is a row of P with +1 on the product; a relative view (“EEM will "
         "beat SPY by 3%”) has +1 and −1. Q holds the values. The uncertainty of each view is")
st.latex(r"\Omega_{kk} = \tau\,(P\Sigma P^\top)_{kk}\,\frac{1-c_k}{c_k}")
st.write("with c the confidence of question QH6 ("
         + ", ".join(f"{k} = {v:.0%}" for k, v in CONFIDENCE_LEVELS.items())
         + "). For an absolute view (PΣPᵀ)ₖₖ = σᵢ²; for a relative view it is the variance of the spread, "
         "σᵢ² + σⱼ² − 2σᵢⱼ. Medium gives the view the same uncertainty as the equilibrium. A view more than two "
         "standard deviations √(τ·PΣPᵀ) away from the equilibrium value Pπ triggers a warning (this is the "
         "uncertainty of the equilibrium return itself: with τ = 0.05, as precise as an average over 20 years). Then we run "
         "mean-variance with μ_BL.")

st.subheader("Minimum variance")
st.write("The portfolio with the lowest volatility inside the client's limits: the same QP with μ = 0. It is the left "
         "end of the efficient frontier, and the limit of mean-variance when γ becomes very large.")

st.subheader("Risk parity / Equal Risk Contribution")
st.latex(r"RC_i = \frac{x_i (\Sigma x)_i}{x^\top \Sigma x} = \frac{1}{n}")
st.write("Does not use expected returns at all: every asset brings the same share of risk. When the client's "
         "limits make equal shares impossible, we take the portfolio inside the limits whose risk shares are "
         "the closest to 1/n.")

st.header("4. Risk measures")
st.write("Beta with the covariance method β = cov(R_p, R_b) / var(R_b) and tracking error = volatility of "
         "R_p − R_b, both against the benchmark chosen by the client (by default: the S&P 500 and "
         "government bonds in the client's currency, with the same share of equities as the recommended portfolio).")

st.header("5. Your plan")
st.write("**Projection.** We simulate 5,000 possible futures of the portfolio, month by month (Monte Carlo). "
         "Each month the return is drawn from a normal distribution with the expected "
         "return of our forecast (market equilibrium, + views if any) minus our fee, and the volatility of the portfolio; "
         "then the monthly contribution is added:")
st.latex(r"W_{t+1} = W_t\,(1 + r_{t+1}) + \text{contribution}, \qquad "
         r"r_{t+1} \sim N\!\left(\tfrac{\mu_p - \text{fee}}{12},\; \tfrac{\sigma_p^2}{12}\right)")
st.write("We use the same forecast for every method: projecting the returns of the last 5 years into the "
         "future would be too optimistic or too pessimistic. The page shows the 5%, 50% and 95% percentiles of the "
         "final wealth, the share of futures that reach the goal, and the share of futures where the portfolio "
         "falls at some point by more than the loss accepted in QE2 (from a previous peak).")
st.write("**Orders.** Amount × weight ÷ last price, rounded down to whole ETF shares (crypto: 0.0001 units); the rest "
         "stays in cash. **Fee.** Our management fee is 0.50% per year; it is also taken in the backtest. "
         "**Report.** A PDF with the profile, limits, portfolio, plan, orders and points to review.")

st.header("6. Backtest with transaction costs")
st.write("Out-of-sample: at each rebalancing we only use the past 60 months. Between rebalancings the weights "
         "drift with the markets. The client chooses the period, the benchmark, the rebalancing frequency, "
         "the cost and a turnover limit.")
st.write("Cost-aware mean-variance: we start from the current weights x⁰ and add what we buy (Δx⁺) "
         "and what we sell (Δx⁻):")
st.latex(r"\min_{x,\Delta x^-,\Delta x^+} \; \tfrac{1}{2} x^\top \Sigma x - \tfrac{1}{\gamma}"
         r"\left(x^\top \mu - c^\top \Delta x^- - c^\top \Delta x^+\right)")
st.latex(r"\text{s.t. } \textstyle\sum x_i + c^\top(\Delta x^- + \Delta x^+) = 1,\quad "
         r"x = x^0 + \Delta x^+ - \Delta x^-,\quad \sum (\Delta x^-_i + \Delta x^+_i) \le \tau^+")
st.write("So the portfolio only trades when the gain is larger than the cost. Resampled MV and risk parity "
         "move towards their new target, but never trade more than the turnover limit τ⁺.")

st.header("7. Stress test")
st.write("Buy and hold from the market peak of 2008, 2020 and 2022, compared with the largest temporary loss "
         "the client said (QE2) he could tolerate without abandoning the strategy. A fund that did not exist yet "
         "at the start of a crisis is replaced by an older fund on the same market (the Swiss equity fund EWL for "
         "the SMI and SMIM funds, the S&P 500 for the world fund, US aggregate bonds for the hedged international "
         "bonds); a fund without such a replacement is counted as cash, and the page says how much.")

st.header("8. Limitations")
st.markdown("""
- Past returns are a noisy guess of future returns (this is why we recommend the market equilibrium returns, and offer resampling).
- 60 months is a short window; a longer one is more stable but reacts slower.
- We do not hedge the currency risk of equities; the CHF risk-free rate is an approximation (SNB policy rate).
- The regional weights of the funds are approximate, rounded index weights, not live data.
- Yahoo Finance does not always adjust the prices of the Swiss bond funds for their coupon payments: their
  measured volatility is a little too high and their past return a little too low.
- One fund cannot be more than 40% of the portfolio, so a client with only two equity funds holds at most 80% in
  equities (about 94% for a client who prefers Europe), even if his profile allows 100%.
- After our fee, a portfolio of short-term bonds earns less than a savings account: the app says so.
- The weights of the market portfolio are our own round numbers, not measured market capitalisations.
- The current month is not finished: it is used as if it was a full month.
- We do not shrink the covariance matrix (Ledoit-Wolf): with about 10 products and 60 months, the estimation
  error of Σ is small compared to the error on μ. With many more products, shrinkage would become useful.
- Bid-ask costs are the same for every product; in reality crypto and small markets cost more.
- The projection assumes normal and independent monthly returns: real markets have more extreme months. The
  fees of the ETFs themselves (TER) and inflation are not included.
- The questionnaire scores, the capacity tiers and the γ anchors are modelling choices: they are calibrated to be
  consistent with QE3, not estimated on investor data.
""")

st.header("Sources")
st.markdown("""
Markowitz (1952) · Black & Litterman (1992) · He & Litterman (1999) ·
Michaud (2007) · Maillard, Roncalli & Teiletche (2010) · Doeswijk, Lam & Swinkels (2014) · Roncalli (2013) ·
QARM II lecture notes and labs (HEC Lausanne)
""")
