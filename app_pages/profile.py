"""Page 1: questionnaire -> risk aversion gamma (tolerance) and allocation limits (capacity)."""
import pandas as pd
import streamlit as st

from src.app_helpers import show_profile_messages
from src.client_profile import GAMMA_MAX, GAMMA_MIN, ClientProfile, gamma_from_rt, qe3_gamma_interval
from src.data import ASSETS
from src.questionnaire import (CAPACITY, CRYPTO_CAP, HORIZON, QE3_PROFILES, RISK_TOLERANCE, UNIVERSE,
                               missing_answers)

st.title("Robo-advisor: your investor profile")
st.write("Two things decide your portfolio: **how much risk you are willing to take** (your risk "
         "tolerance, which sets γ) and **how much risk you can afford** (your financial situation, "
         "horizon and liquidity needs, which set the allocation limits).")

saved = st.session_state["profile"].answers if "profile" in st.session_state else {}


def ask(code, text, opts):
    """Radio question; returns the letter of the answer (or None)."""
    letters = list(opts)
    previous = saved.get(code)
    return st.radio(f"**{text}**", letters, format_func=lambda x: opts[x][0],
                    index=letters.index(previous) if previous in letters else None, key=f"q_{code}")


answers = {}
with st.form("questionnaire"):
    st.header("1. Your attitude to risk")
    for code, (text, opts) in RISK_TOLERANCE.items():
        answers[code] = ask(code, text, opts)

    st.header("2. Your horizon and liquidity needs")
    for code, (text, opts) in HORIZON.items():
        answers[code] = ask(code, text, opts)

    st.header("3. Your financial situation")
    for code, (text, opts) in CAPACITY.items():
        answers[code] = ask(code, text, opts)
    st.caption("Emergency savings: exclude this investment and money already committed to a known payment. "
               "Manageable commitments: you can pay them without borrowing more or selling essential assets.")

    st.header("4. What you want to invest in")
    text, opts = UNIVERSE
    answers["QF1"] = st.multiselect(f"**{text}**", list(opts), default=saved.get("QF1", list(opts)),
                                    format_func=lambda x: opts[x][0], key="q_QF1")
    text, opts = CRYPTO_CAP
    answers["QF2"] = ask("QF2", text + " (only if you selected cryptoassets)", opts)

    submitted = st.form_submit_button("Compute my profile", type="primary")

if submitted:
    missing = missing_answers(answers)
    if missing:
        st.error("Please answer every question (missing: " + ", ".join(missing) + ").")
    else:
        st.session_state["profile"] = ClientProfile.from_answers(answers)
        # new profile -> we start again from our recommended products and constraints
        for key in ["assets", "constraints"] + [k for k in st.session_state if k.startswith("pick_")]:
            st.session_state.pop(key, None)

# ---------- Show profile (stays visible for the next pages) ----------
if "profile" in st.session_state:
    profile = st.session_state["profile"]
    c = profile.recommended_constraints
    st.divider()
    st.header(f"Your profile: {profile.label} tolerance, {profile.capacity_label.lower()} capacity")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Risk tolerance (RT)", f"{profile.rt:.1f} / 100")
    col2.metric("Risk aversion γ", f"{profile.gamma:.2f}")
    col3.metric("Financial capacity", f"{profile.capacity_label} ({profile.capacity_score}/9)")
    col4.metric("Maximum in equities", f"{c['max_equity']:.0%}")
    show_profile_messages(profile, notes=True)

    st.subheader("Your allocation limits")
    st.table(pd.DataFrame({
        "Limit": ["Maximum in equities", "Minimum in bonds", "Maximum in gold and commodities",
                  "Maximum in crypto", "Maximum per product"],
        "Value": [f"{c[k]:.0%}" for k in ("max_equity", "min_bonds", "max_other", "max_crypto",
                                          "max_weight_per_asset")],
        "Set by": [", ".join(profile.limit_sources["max_equity"]),
                   ", ".join(profile.limit_sources["min_bonds"]) or "no minimum",
                   f"financial capacity ({profile.capacity_label})",
                   "your choice and financial capacity", "diversification rule"],
    }))
    st.write("Recommended products: " + ", ".join(f"{t} ({ASSETS[t][0]})" for t in profile.recommended_assets))

    with st.expander("How is γ computed?"):
        st.write("Your risk tolerance RT is the average of your four answers in part 1 (each scored from 0 to 100). "
                 "It is converted into the risk aversion γ used by the mean-variance optimiser:")
        st.latex(rf"\gamma(RT) = {GAMMA_MAX:g} \times \left(\frac{{{GAMMA_MIN:g}}}{{{GAMMA_MAX:g}}}\right)^{{RT/100}}")
        st.write("The two anchors are calibrated on question QE3. The optimiser maximises "
                 "U = μ − γ/2 · σ², so between two options (μ₁, σ₁) and (μ₂, σ₂) a client is indifferent when "
                 "γ = 2(μ₂ − μ₁) / (σ₂² − σ₁²). Each QE3 answer therefore reveals an interval of γ, and the "
                 "formula above puts the score of each answer inside its interval:")
        opts = RISK_TOLERANCE["QE3"][1]
        rows = []
        for letter, (mu, sigma) in QE3_PROFILES.items():
            low, high = qe3_gamma_interval(letter)
            score = opts[letter][1]
            rows.append({"QE3 answer": opts[letter][0], "RT score": score, "γ(score)": round(gamma_from_rt(score), 2),
                         "Revealed γ interval": f"{low:.2f} – {'∞' if high == float('inf') else f'{high:.2f}'}"})
        st.table(pd.DataFrame(rows))
        st.write("Your financial situation does **not** change γ: it sets the allocation limits above. "
                 "This way your tolerance always matters, and your capacity is not counted twice.")

    st.page_link("app_pages/choices.py", label="Next: choose your products and constraints", icon="➡️")
