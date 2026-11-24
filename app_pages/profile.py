"""Page 1: questionnaire -> risk aversion gamma (tolerance) and allocation limits (capacity)."""
import pandas as pd
import streamlit as st

from src.app_helpers import get_currency, recommendation_summary, refresh_profile, reset_choices, show_profile_messages
from src.client_profile import GAMMA_MAX, GAMMA_MIN, ClientProfile, gamma_from_rt, qe3_gamma_interval
from src.data import ASSETS, available_assets
from src.questionnaire import (CAPACITY, CRYPTO_CAP, CRYPTO_CHOICE, EXPERIENCE, HOME_BIAS, HORIZON, QE3_PROFILES,
                               REGION_QUESTION, RISK_TOLERANCE, UNIVERSE, VIEWS_QUESTION, is_experienced,
                               missing_answers)
from src.style import nav_buttons, page_header

page_header(1, "Your investor profile",
            "Two things decide your portfolio: **how much risk you are willing to take** (your risk tolerance, which "
            "sets γ) and **how much risk you can afford** (your financial situation, horizon and liquidity needs, "
            "which set the allocation limits).")

saved = st.session_state["profile"].answers if "profile" in st.session_state else {}
client = st.session_state.get("client_number", 0)  # changes when "New client" is clicked: the questions are new


def ask(code, text, opts):
    """Radio question; returns the letter of the answer (or None)."""
    letters = list(opts)
    previous = saved.get(code)
    return st.radio(f"**{text}**", letters, format_func=lambda x: opts[x][0],
                    index=letters.index(previous) if previous in letters else None, key=f"q_{code}_{client}")


def questionnaire():
    """The questions and the button that computes the profile."""
    answers = {}
    # No st.form: the conditional questions (crypto cap, market views) appear as soon as they apply.
    with st.container(border=True):
        st.subheader("1. Your attitude to risk")
        for code, (text, opts) in RISK_TOLERANCE.items():
            answers[code] = ask(code, text, opts)

    with st.container(border=True):
        st.subheader("2. Your horizon and liquidity needs")
        for code, (text, opts) in HORIZON.items():
            answers[code] = ask(code, text, opts)

    with st.container(border=True):
        st.subheader("3. Your financial situation")
        for code, (text, opts) in CAPACITY.items():
            answers[code] = ask(code, text, opts)
        st.caption("Emergency savings: exclude this investment and money already committed to a known payment. "
                   "Manageable commitments: you can pay them without borrowing more or selling essential assets.")

    with st.container(border=True):
        st.subheader("4. What you want to invest in")
        text, opts = UNIVERSE
        answers["QF1"] = st.multiselect(f"**{text}**", list(opts), default=saved.get("QF1", [o for o in opts if o != CRYPTO_CHOICE]),
                                        format_func=lambda x: opts[x][0], key=f"q_QF1_{client}")
        if CRYPTO_CHOICE in answers["QF1"]:
            text, opts = CRYPTO_CAP
            answers["QF2"] = ask("QF2", text, opts)
        text, opts = REGION_QUESTION
        answers["QF3"] = ask("QF3", text, opts)
        st.caption(f"Your region gets at least {HOME_BIAS:.0%} of your equities; the rest stays worldwide, because one "
                   "region alone is less diversified. Bonds and cash are mainly in the currency of your portfolio (menu "
                   "on the left), because they are its safe part.")

    with st.container(border=True):
        st.subheader("5. Your experience")
        for code, (text, opts) in EXPERIENCE.items():
            answers[code] = ask(code, text, opts)
        if is_experienced(answers):
            st.caption("You qualify as an experienced investor: you can add your own market views (Black-Litterman "
                       "model, on the portfolio page).")
            text, opts = VIEWS_QUESTION
            answers["QH4"] = ask("QH4", text, opts)

    if st.button("Compute my profile", type="primary", icon=":material/calculate:"):
        missing = missing_answers(answers)
        if missing:
            st.error("Please answer every question (missing: " + ", ".join(missing) + ").")
        else:
            st.session_state["profile"] = ClientProfile(answers, get_currency())
            reset_choices()  # new profile -> we start again from our recommended products, limits and method
            # New version: the box "Change your answers" closes, so the client sees his new result
            st.session_state["profile_version"] = st.session_state.get("profile_version", 0) + 1
            st.rerun()


def show_result():
    """The client's profile, his limits and our recommendation."""
    profile = refresh_profile()
    if st.session_state.get("profile_version", 0) != st.session_state.get("profile_shown"):  # just (re)computed
        st.session_state["profile_shown"] = st.session_state.get("profile_version", 0)
        st.toast("Your profile is ready.", icon=":material/check_circle:")
    c = profile.recommended_constraints
    st.caption("RESULT")
    st.header(f"{profile.label} tolerance, {profile.capacity_label.lower()} financial capacity")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Risk tolerance (RT)", f"{profile.rt:.1f} / 100")
    col2.metric("Risk aversion γ", f"{profile.gamma:.2f}")
    col3.metric("Financial capacity", profile.capacity_label,
                help=f"Score {profile.capacity_score}/9 (Low 0–2, Limited 3–4, Moderate 5–6, High 7–9)"
                     + (". Lowered to Limited because of low savings or heavy commitments." if profile.safeguard else "."))
    col4.metric("Maximum in equities", f"{c['max_equity']:.0%}")
    tickers = [t for t in profile.recommended_assets if t in available_assets()]
    summary = recommendation_summary(profile, tickers, c)
    if summary:
        with st.container(border=True):
            st.markdown("**Our recommendation**")
            st.write(summary)
    show_profile_messages(profile, notes=True)
    st.caption("Experienced investor: " + ("yes" if profile.experienced else "no")
               + (" · Your own market views: yes (Black-Litterman, on the portfolio page)" if profile.wants_views
                  else " · Your own market views: no"))

    st.subheader("Your allocation limits")
    st.dataframe(pd.DataFrame({
        "Limit": ["Maximum in equities", "Minimum in bonds", "Minimum in short-term government bonds",
                  "Maximum in gold and commodities", "Maximum in crypto", "Maximum per product",
                  f"Minimum share of your equities in {profile.region}", "Largest loss in a bad year (5% chance)"],
        "Value": [f"{c[k]:.0%}" for k in ("max_equity", "min_bonds", "min_short_term", "max_other", "max_crypto",
                                          "max_weight_per_asset", "min_home_equity")]
        + ["no limit" if profile.max_loss is None else f"{-profile.max_loss:.0%}"],
        "Set by": [", ".join(profile.limit_sources["max_equity"]),
                   ", ".join(profile.limit_sources["min_bonds"]) or "no minimum",
                   ", ".join(profile.limit_sources["min_short_term"]) or "no money needed soon",
                   f"financial capacity ({profile.capacity_label})",
                   "you did not choose crypto" if CRYPTO_CHOICE not in profile.answers["QF1"]
                   else "your horizon or financial capacity" if profile.crypto_blocked
                   else "your choice (QF2) and financial capacity",
                   "diversification rule (short-term bonds: no maximum, narrow products: 15%)",
                   "your region (QF3)" if profile.region != "World" else "no regional preference",
                   "you accept more than 30% and your capacity is high" if profile.max_loss is None
                   else "the loss " + profile.loss_reason],
    }), hide_index=True)
    st.write("**Recommended products:** " + ", ".join(f"{t} ({ASSETS[t][0]})" for t in profile.recommended_assets))

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
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.write("Your financial situation does **not** change γ: it sets the allocation limits above. "
                 "This way your tolerance always matters, and your capacity is not counted twice.")


# Once the profile is computed, the result comes first and the questions can still be changed below it
if "profile" in st.session_state:
    show_result()
    with st.expander("Change your answers", icon=":material/edit:",
                     key=f"change_answers_{st.session_state.get('profile_version', 0)}"):
        questionnaire()
    nav_buttons("app_pages/welcome.py", "app_pages/choices.py", "Products and limits")
else:
    questionnaire()
