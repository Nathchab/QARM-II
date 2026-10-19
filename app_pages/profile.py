"""Adaptive questionnaire with provisional automatic gamma; optimizers unchanged."""
from copy import deepcopy
from dataclasses import asdict

import pandas as pd
import streamlit as st

from src.client_profile import ClientProfile
from src.app_helpers import ACADEMIC_NOTICE, show_profile_notice
from src.gamma_policy import DEFAULT_RESEARCH_MAPPING, REQUIRED_FINANCIAL_INPUTS, effective_constraints
from src.financial_capacity import FINANCIAL_FIELDS, POLICY_VERSION
from src.questionnaire_definitions import QUESTIONS, SECTIONS
from src.questionnaire_routing import applicable_questions
from src.questionnaire_scoring import QuestionnaireError, validate_answers
from src.questionnaire_state import initialize_draft, publish_profile, save_widget


initialize_draft(st.session_state)
answers = st.session_state["questionnaire_answers"]
capacity = st.session_state["capacity_answers"]
st.title("Robo-advisor: your investor profile")
st.caption("QARM II · Adaptive questionnaire · Scores remain separate from financial capacity and optimization γ")
st.caption(ACADEMIC_NOTICE)
st.info("RT determines provisional willingness gamma automatically. Other questionnaire dimensions remain "
        "separate. Requested questionnaire portfolio settings and routes are recorded only.")


def persist(store, answer_key, widget_key):
    save_widget(st.session_state, store, answer_key, widget_key)


def number_detail(answer_key, label, minimum=None, maximum=None):
    key = f"questionnaire_widget_{answer_key}"
    if key not in st.session_state:
        st.session_state[key] = answers.get(answer_key)
    st.number_input(label, min_value=minimum, max_value=maximum, value=None, key=key,
                    on_change=persist, args=("questionnaire_answers", answer_key, key))


def text_detail(answer_key, label):
    key = f"questionnaire_widget_{answer_key}"
    if key not in st.session_state:
        st.session_state[key] = answers.get(answer_key, "")
    st.text_input(label, key=key, on_change=persist, args=("questionnaire_answers", answer_key, key))


def render_question(code):
    q = QUESTIONS[code]
    key = f"questionnaire_widget_{code}"
    if key not in st.session_state:
        st.session_state[key] = deepcopy(answers.get(code, [] if q.multi else None))
    label = lambda option: q.options[option].text
    options = list(q.options)
    if q.multi:
        st.multiselect(f"{code}. {q.text}", options, format_func=label, key=key,
                       on_change=persist, args=("questionnaire_answers", code, key))
        if code in ("QG1", "QJ2") and not st.session_state[key]:
            if st.button("Confirm no selections", key=f"confirm_empty_{code}"):
                answers[code] = []
                st.rerun()
        if code in ("QB2", "QC4"):
            st.caption("Approved addition: ‘None of these’ = 0. Leaving this unanswered does not score zero.")
    else:
        st.radio(f"{code}. {q.text}", options, index=None, format_func=label, key=key,
                 on_change=persist, args=("questionnaire_answers", code, key))
    if q.note:
        st.caption(q.note)
    if code == "QF4" and answers.get(code) == "d":
        number_detail("QF4_limit", "Custom maximum per security (%)", minimum=0.000001, maximum=100.0)
    elif code == "QH2" and answers.get(code) == "d":
        text_detail("QH2_custom", "Custom benchmark name / identifier")
    elif code == "QH3" and answers.get(code) == "d":
        number_detail("QH3_limit", "Selected maximum annualized tracking error (%)", minimum=15.0)
    elif code == "QF5" and answers.get(code) in ("b", "c"):
        text_detail("QF5_details", "Requested gross exposure / long and short limits (recorded only)")
    elif code == "QF6" and answers.get(code) == "d":
        text_detail("QF6_details", "Custom institutional leverage limit (recorded only)")
    elif code == "QJ2" and answers.get(code):
        text_detail("QJ2_details", "Policy values / custom constraint description (recorded only)")


def save_view(index, field, widget_key):
    st.session_state["questionnaire_answers"]["QH5"][index][field] = st.session_state[widget_key]


def render_views():
    st.subheader(f"QH5. {QUESTIONS['QH5'].text}")
    st.caption("Views and confidence are recorded only. Relative views and 100% confidence are not sent "
               "to the current Black–Litterman solver.")
    views = answers.setdefault("QH5", [])
    if st.button("Add a quantitative view"):
        views.append({"asset": "", "value": None, "confidence": None, "measure": "return"})
        st.rerun()
    for i, view in enumerate(views):
        st.markdown(f"**View {i + 1}**")
        fields = [("asset", "Asset / basket identifier", "text")]
        if answers.get("QH4") == "c":
            fields.append(("relative_to", "Comparison asset / basket identifier", "text"))
        else:
            fields.append(("measure", "Absolute expected return / premium", "measure"))
        fields += [("value", "Annual expected return / premium / outperformance (%)", "number"),
                   ("confidence", f"QH6. {QUESTIONS['QH6'].text}", "confidence")]
        for field, label, kind in fields:
            key = f"questionnaire_view_{i}_{field}"
            if key not in st.session_state:
                st.session_state[key] = view.get(field, "" if kind == "text" else None)
            kwargs = dict(key=key, on_change=save_view, args=(i, field, key))
            if kind == "text":
                st.text_input(label, **kwargs)
            elif kind == "number":
                st.number_input(label, value=None, **kwargs)
            elif kind == "measure":
                st.radio(label, ["return", "premium"], index=None, **kwargs)
            else:
                st.radio(label, list(QUESTIONS["QH6"].options), index=None,
                         format_func=lambda item: QUESTIONS["QH6"].options[item].text, **kwargs)
        if st.button("Remove this view", key=f"remove_view_{i}"):
            views.pop(i)
            for key in list(st.session_state):
                if key.startswith("questionnaire_view_"):
                    del st.session_state[key]
            st.rerun()


advanced_key = "questionnaire_widget_advanced_enabled"
if advanced_key not in st.session_state:
    st.session_state[advanced_key] = answers.get("advanced_enabled", False)
st.checkbox("Include the optional advanced questions", key=advanced_key, on_change=persist,
            args=("questionnaire_answers", "advanced_enabled", advanced_key),
            help="Explicit opt-in for ambiguous advanced eligibility: short selling, leverage, "
                 "optional market views, alternative-assets and scenario sensitivity. This does not raise RT.")

active = applicable_questions(answers)
errors = validate_answers(answers)
incomplete = {key.split("[")[0].split("_")[0] for key in errors}
completed = sum(code not in incomplete for code in active)
st.progress(completed / len(active), text=f"{completed}/{len(active)} applicable questions complete")
sections = [section for section in SECTIONS if any(code[1] == section for code in active)] + ["Capacity", "Results"]
current = st.session_state.get("questionnaire_section", sections[0])
if current not in sections:
    current = sections[0]
if st.session_state.get("questionnaire_section_selector") not in sections:
    st.session_state["questionnaire_section_selector"] = current
section = st.selectbox("Questionnaire section", sections,
                       format_func=lambda value: f"{value}. {SECTIONS[value]}" if value in SECTIONS else value,
                       key="questionnaire_section_selector")
st.session_state["questionnaire_section"] = section

if section in SECTIONS:
    st.header(f"{section}. {SECTIONS[section]}")
    for code in active:
        if code[1] != section:
            continue
        if code == "QH5":
            render_views()
        elif code != "QH6":
            render_question(code)
elif section == "Capacity":
    st.header("Financial capacity · four-question research policy")
    st.write("Age, income stability, emergency savings and commitments give a provisional score from 0 to 9. "
             "Your original QD1/QD2 answers supply horizon and liquidity information without repeated questions. "
             "QE2 remains a willingness measure, not financial loss-bearing capacity.")
    for name, (label, options) in FINANCIAL_FIELDS.items():
        key = f"questionnaire_capacity_{name}"
        if key not in st.session_state:
            st.session_state[key] = capacity.get(name)
        st.selectbox(label, list(options), index=None, key=key, on_change=persist,
                     args=("capacity_answers", name, key))
    st.caption("Emergency savings: exclude this investment account, new borrowing and money already committed "
               "to a known payment. Exactly 3 or 6 months belongs to 3–6 months. Essential expenses include regular debt payments.")
    st.caption("Commitments: consider debt payments, dependants and unavoidable obligations. Manageable means "
               "payments can currently be met without further borrowing or selling essential assets.")
    for code in ("QD1", "QD2"):
        if answers.get(code) in QUESTIONS[code].options:
            st.write(f"{code} (reused): {QUESTIONS[code].options[answers[code]].text}")
    st.caption("Capacity floor: score 0–3 → 12; 4–6 → 6; 7–9 → 2. Low emergency savings or significant "
               "commitments impose a minimum floor of 6. These are research restrictions, not a monetary loss budget.")
    st.subheader("Automatic research gamma")
    st.write("Willingness gamma is calculated from RT. Execution gamma = max(willingness gamma, "
             "capacity floor). No investor gamma entry is required; HZ does not adjust gamma.")
    with st.expander("Provisional research mapping assumptions"):
        st.write("gW = g_lowRT × (g_highRT / g_lowRT)^((RT − RT_min)/(RT_max − RT_min)). "
                 "These illustrative study anchors are not empirically validated preferences.")
        st.json(asdict(DEFAULT_RESEARCH_MAPPING))
else:
    st.header("Review and compute your profile")
    if errors:
        st.write("Complete these applicable questionnaire inputs:")
        for code, error in errors.items():
            st.write(f"• {code}: {error}")
    if st.button("Compute my profile", type="primary"):
        try:
            profile = ClientProfile.from_financial_questionnaire(deepcopy(answers), deepcopy(capacity))
        except (QuestionnaireError, ValueError) as error:
            st.error(str(error))
        else:
            old = st.session_state.get("profile")
            if old is not None and getattr(old, "capacity_policy_version", None) != POLICY_VERSION:
                st.session_state["capacity_migration_candidate"] = profile
                st.info("Review the policy migration below. Your saved legacy profile remains active until you apply it.")
            else:
                publish_profile(st.session_state, profile)
                st.success("Research profile saved. Product defaults and optional allocation choices have been reset.")

    pending = st.session_state.get("capacity_migration_candidate")
    if pending is not None:
        old = st.session_state["profile"]
        stale = (pending.questionnaire_answers != answers or any(
            capacity.get(key) != pending.capacity_answers[key] for key in FINANCIAL_FIELDS))
        st.subheader("Review saved-profile migration")
        previous = st.session_state.get("constraints", old.recommended_constraints)
        if getattr(old, "gamma_automatic", False):
            previous = effective_constraints(old.mandatory_constraints, previous)
        rows = [
            {"Policy item": "Capacity score", "Saved policy": f"{old.capacity_score}/13", "New policy": f"{pending.capacity_score}/9"},
            {"Policy item": "Capacity floor", "Saved policy": f"{old.gamma_capacity:g}", "New policy": f"{pending.gamma_capacity:g}"},
            {"Policy item": "Willingness / baseline gamma", "Saved policy": f"{old.gamma_willingness:.5f}", "New policy": f"{pending.gamma_willingness:.5f}"},
            {"Policy item": "Final gamma", "Saved policy": f"{old.gamma:.5f}", "New policy": f"{pending.gamma:.5f}"},
        ]
        rows += [{"Policy item": key, "Saved policy": f"{previous[key]:.0%}", "New policy": f"{value:.0%}"}
                 for key, value in pending.recommended_constraints.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.warning("Applying the new version explicitly retires the legacy acceptable-loss-dependent limits. "
                   "Limits may tighten or loosen. The old policy, answers and optional choices will be archived; "
                   "current product/constraint selections will reset to the new defaults. QE2 is not reused as financial capacity.")
        if stale:
            st.warning("The draft changed after this preview. Compute the profile again before applying migration.")
        if st.button("Apply reviewed policy migration", disabled=stale):
            publish_profile(st.session_state, pending, migrate=True)
            st.rerun()


def navigate(target):
    st.session_state["questionnaire_section_selector"] = target


left, right = st.columns(2)
index = sections.index(section)
left.button("Previous section", disabled=index == 0, on_click=navigate, args=(sections[max(0, index - 1)],))
right.button("Next section", disabled=index == len(sections) - 1, on_click=navigate,
             args=(sections[min(len(sections) - 1, index + 1)],))

if "profile" in st.session_state:
    profile = st.session_state["profile"]
    st.divider()
    if not hasattr(profile, "scores"):
        st.info("Your saved legacy profile remains available. Complete this questionnaire to calculate the new scores.")
    else:
        if profile.questionnaire_answers != answers or any(
                capacity.get(key) != value for key, value in profile.capacity_answers.items()
                if key in FINANCIAL_FIELDS):
            st.caption("Draft answers differ from your saved profile. Compute your profile again to apply them.")
        st.header(f"Risk Tolerance: {profile.risk_classification}")
        st.dataframe(pd.DataFrame([
            {"Dimension": code, "Score / 100": value,
             "Interpretation": profile.dimension_labels.get(code, "Per-view conviction confidence")}
            for code, value in profile.scores.items() if value is not None
        ]), hide_index=True)
        cols = st.columns(4)
        cols[0].metric("RT (original score)", f"{profile.rt:g}")
        automatic = getattr(profile, "gamma_automatic", False)
        cols[1].metric("Willingness gamma" if automatic else "Legacy explicit baseline", f"{profile.gamma_willingness:.2f}")
        cols[2].metric("Capacity floor", f"{profile.gamma_capacity:g}")
        cols[3].metric("Execution gamma (gamma-based methods)", f"{profile.gamma:.2f}")
        if getattr(profile, "capacity_policy_version", None) == POLICY_VERSION:
            st.write(f"Financial capacity: {profile.capacity_score}/9.")
            if profile.capacity_safeguards:
                st.caption("A financial-capacity safeguard applies.")
        else:
            st.info("Saved legacy capacity policy remains active. The four-question draft has not silently changed its limits.")
        show_profile_notice(profile)
        st.caption("Risk Parity does not use gamma in its objective.")
        st.write("Raw timing/withdrawal answers (separate from HZ):")
        st.dataframe(pd.DataFrame([
            {"Question": code, "Raw response": profile.questionnaire_answers[code],
             "Meaning": QUESTIONS[code].options[profile.questionnaire_answers[code]].text}
            for code in ("QD1", "QD2")
        ]), hide_index=True)
        st.caption("These categories do not quantify liabilities or establish liquidity compatibility. "
                   "No horizon adjustment or RT/HZ average is applied to gamma.")
        with st.expander("Advanced details"):
            st.write(f"Willingness gamma: {profile.gamma_willingness:.5f}; capacity floor: {profile.gamma_capacity:g}; "
                     f"final gamma: {profile.gamma:.5f}.")
            st.write(getattr(profile, "gamma_binding_explanation", profile.gamma_source))
            st.write("Portfolio route candidates (recorded only): " + "; ".join(profile.portfolio_routes))
            st.write("Illustrative risk matches: " + (", ".join(profile.illustrative_risk_matches) or "none"))
            if automatic:
                st.json(asdict(profile.gamma_mapping_assumptions))
            st.write("Score calculation details")
            st.json(profile.score_subscores)
            st.write("Mandatory research limits used downstream")
            st.json(profile.recommended_constraints)
            st.write(", ".join(profile.recommended_assets))
        if not automatic:
            st.info("This saved profile uses a legacy explicit baseline. Recompute to use automatic research gamma.")
        with st.expander("Financial inputs still required for Alternative B"):
            for item in REQUIRED_FINANCIAL_INPUTS:
                st.write("• " + item)
            st.write("A justified loss measure, assessment horizon, stress/tail models and policy thresholds "
                     "also need to be defined and validated. No loss budget is inferred here.")
        if getattr(profile, "migration_history", []):
            with st.expander("Archived previous policy"):
                st.json(profile.migration_history)
        st.page_link("app_pages/choices.py", label="Continue with existing products and constraints", icon="➡️")
