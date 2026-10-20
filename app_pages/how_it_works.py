"""Page 6: short explanation of the methods, data and limitations (useful for questions)."""
import streamlit as st

from src.black_litterman import CONFIDENCE_LEVELS, DELTA, TAU
from src.client_profile import GAMMA_MAX, GAMMA_MIN
from src.data import ASSETS

st.title("How it works")

st.header("1. Your profile")
st.write("The questionnaire (QARM II questionnaire, sections D, E and F) measures two different things, "
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
  capacity at Limited. Each tier gives a maximum in equities, a minimum in bonds, and maximums in gold/commodities
  and crypto. The horizon (QD1) and the possible withdrawals (QD2) give their own equity maximum and bond minimum.
  **The most restrictive source applies.** The client can tighten these limits, never relax them.
- **Why capacity does not change γ.** If capacity also raised γ, it would be counted twice (limits *and* a γ
  floor), and for most clients the risk tolerance would no longer change anything. Here a tolerant client with
  a low capacity gets the most dynamic portfolio *inside* the limits his situation allows.
- **Warnings** are kept for material facts only: emergency savings below 3 months, significant commitments,
  money needed within a year, or likely withdrawals above 25%.
- **Products.** The asset classes chosen in QF1 give the recommended products; crypto (QF2) is also capped by
  capacity. If the chosen classes cannot satisfy the limits (for example no bond product but a bond minimum), we
  add the missing products and say so.
- **Risk parity** does not use γ: only the covariance and the allocation limits.
""")

st.header("2. Data")
st.write("The app downloads the daily prices directly from Yahoo Finance when it starts (again every day, or "
         "with the button 'Update data now'). Nothing has to be run before. We use monthly returns; "
         "expected returns and covariances are estimated on the last 60 months. In CHF, prices are converted "
         "with the USD/CHF rate.")
st.table({"Ticker": list(ASSETS), "Product": [a[0] for a in ASSETS.values()],
          "Class": [a[1] for a in ASSETS.values()]})

st.header("3. Methods")
st.subheader("Mean-variance (Markowitz, lab LN1)")
st.latex(r"\min_x \; \tfrac{1}{2} x^\top \Sigma x - \tfrac{1}{\gamma} x^\top \mu "
         r"\quad \text{s.t. } \textstyle\sum x_i = 1,\; 0 \le x_i \le x^+,\; \text{class constraints}")
st.write("Same QP as in the lab, where the lab's γ is the risk tolerance = 1 / our risk aversion.")

st.subheader("Resampled mean-variance (Michaud 2007, lab LN3)")
st.write("Expected returns are very uncertain, so mean-variance weights are unstable. We simulate 100 new "
         "samples of 60 months from N(μ, Σ), optimise each one and average the weights.")

st.subheader("Black-Litterman (1992)")
st.latex(r"\pi = r_f + \delta \Sigma w_{mkt}")
st.latex(r"\mu_{BL} = \left[(\tau\Sigma)^{-1} + P^\top \Omega^{-1} P\right]^{-1}"
         r"\left[(\tau\Sigma)^{-1}\pi + P^\top \Omega^{-1} Q\right]")
st.write(f"π: equilibrium returns of the world market portfolio (δ = {DELTA}, τ = {TAU}). "
         "Q: your views, P: which asset each view is about, Ω: uncertainty of the views "
         "(Ω = τσ²(1−c)/c, with c your confidence from question QH6: "
         + ", ".join(f"{k} = {v:.0%}" for k, v in CONFIDENCE_LEVELS.items())
         + "). Medium gives the view the same uncertainty as the equilibrium. Then we run mean-variance with μ_BL.")

st.subheader("Risk parity / Equal Risk Contribution")
st.latex(r"RC_i = \frac{x_i (\Sigma x)_i}{x^\top \Sigma x} = \frac{1}{n}")
st.write("Does not use expected returns at all: every asset brings the same share of risk.")

st.header("4. Risk measures (lab LN2)")
st.write("Beta with the covariance method β = cov(R_p, R_b) / var(R_b) and tracking error = volatility of "
         "R_p − R_b, both against the benchmark chosen by the client (by default: SPY up to the client's equity maximum, the rest in AGG).")

st.header("5. Backtest with transaction costs (LN3)")
st.write("Out-of-sample: at each rebalancing we only use the past 60 months. Between rebalancings the weights "
         "drift with the markets. The client chooses the period, the benchmark, the rebalancing frequency, "
         "the cost and a turnover limit.")
st.write("Cost-aware mean-variance (LN3): we start from the current weights x⁰ and add what we buy (Δx⁺) "
         "and what we sell (Δx⁻):")
st.latex(r"\min_{x,\Delta x^-,\Delta x^+} \; \tfrac{1}{2} x^\top \Sigma x - \tfrac{1}{\gamma}"
         r"\left(x^\top \mu - c^\top \Delta x^- - c^\top \Delta x^+\right)")
st.latex(r"\text{s.t. } \textstyle\sum x_i + c^\top(\Delta x^- + \Delta x^+) = 1,\quad "
         r"x = x^0 + \Delta x^+ - \Delta x^-,\quad \sum (\Delta x^-_i + \Delta x^+_i) \le \tau^+")
st.write("So the portfolio only trades when the gain is larger than the cost. Resampled MV and risk parity "
         "move towards their new target, but never trade more than the turnover limit τ⁺.")

st.header("6. Time machine")
st.write("Buy and hold from the market peak of 2008, 2020 and 2022, compared with the largest temporary loss "
         "the client said (QE2) he could tolerate without abandoning the strategy.")

st.header("7. Limitations")
st.markdown("""
- Past returns are a noisy guess of future returns (this is why we offer Black-Litterman and resampling).
- 60 months is a short window; a longer one is more stable but reacts slower.
- In CHF, we do not hedge the currency risk; the CHF risk-free rate is an approximation (SNB policy rate).
- Market weights for Black-Litterman are rounded approximations.
- We do not shrink the covariance matrix (Ledoit-Wolf): with about 10 products and 60 months, the estimation
  error of Σ is small compared to the error on μ. With many more products, shrinkage would become useful.
- Bid-ask costs are the same for every product; in reality crypto and small markets cost more.
- The questionnaire scores, the capacity tiers and the γ anchors are modelling choices: they are calibrated to be
  consistent with QE3, not estimated on investor data.
""")

st.header("Sources")
st.markdown("""
Markowitz (1952) · Black & Litterman (1992) · He & Litterman (1999) ·
Michaud (2007) · Maillard, Roncalli & Teiletche (2010) · Doeswijk, Lam & Swinkels (2014) · Roncalli (2013) ·
QARM II lecture notes and labs (HEC Lausanne)
""")
