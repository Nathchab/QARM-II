"""Page 2: the client picks the products and the constraints. We propose our recommendation."""
import streamlit as st

from src.app_helpers import get_profile, show_profile_notice
from src.data import ASSETS, available_assets, first_date
from src.gamma_policy import effective_constraints
from src.optimizer import check_constraints
from src.portfolios import WINDOW
from src.portfolio_policy import allocation_feasibility

profile = get_profile()
rec_assets = profile.recommended_assets
rec_c = profile.recommended_constraints

st.title("Your products and constraints")
automatic = getattr(profile, "gamma_automatic", False)
if automatic:
    st.write("Products start from prototype defaults. Optional allocation preferences can tighten the "
             "mandatory prototype limits. Requests to relax them are recorded but do not change the "
             "limits used for execution. Feasibility does not establish financial suitability.")
else:
    st.write("Legacy mode permits allocation edits with warnings. These optional edits do not "
             "establish that mandatory suitability requirements have been met.")

if st.button("↩️ Back to our recommendation"):
    for key in ["assets", "constraints"] + [k for k in st.session_state if k.startswith("pick_")]:
        st.session_state.pop(key, None)
    st.rerun()

saved_assets = st.session_state.get("assets", rec_assets)
saved_c = st.session_state.get("constraints", rec_c)

# ---------- Products ----------
st.header("1. Products")
label = lambda t: f"{t} – {ASSETS[t][0]} (data since {first_date(t):%Y})"
tickers = []
for asset_class in ["Equity", "Bond", "Other", "Crypto"]:
    options = [t for t in available_assets() if ASSETS[t][1] == asset_class]
    tickers += st.multiselect(asset_class, options, default=[t for t in saved_assets if t in options],
                              format_func=label, key=f"pick_{asset_class}")

added = [t for t in tickers if t not in rec_assets]
removed = [t for t in rec_assets if t not in tickers]
if added or removed:
    st.caption(f"Compared with our recommendation: added {', '.join(added) or 'nothing'}, "
               f"removed {', '.join(removed) or 'nothing'}.")
if tickers:
    youngest = max(tickers, key=first_date)
    st.caption(f"The youngest product is {youngest} ({first_date(youngest):%B %Y}). We need {WINDOW} months "
               f"of history to estimate the inputs, so the backtest can start 5 years later.")

# ---------- Constraints ----------
st.header("2. Constraints")
col1, col2 = st.columns(2)
max_equity = col1.slider("Maximum in equities (%)", 0, 100, int(saved_c["max_equity"] * 100), 5,
                         key="pick_max_equity", help=f"Recommended: {rec_c['max_equity']:.0%}")
min_bonds = col2.slider("Minimum in bonds (%)", 0, 100, int(saved_c["min_bonds"] * 100), 5,
                        key="pick_min_bonds", help=f"Recommended: {rec_c['min_bonds']:.0%}")
max_other = col1.slider("Maximum in gold and commodities (%)", 0, 100, int(saved_c["max_other"] * 100), 5,
                        key="pick_max_other", help=f"Recommended: {rec_c['max_other']:.0%}")
max_crypto = col2.slider("Maximum in crypto (%)", 0, 30, int(saved_c["max_crypto"] * 100), 1,
                         key="pick_max_crypto", help=f"Recommended: {rec_c['max_crypto']:.0%}")
max_weight = col1.slider("Maximum per product (%)", 5, 100, int(saved_c["max_weight_per_asset"] * 100), 5,
                         key="pick_max_weight", help=f"Recommended: {rec_c['max_weight_per_asset']:.0%}")

c = {"max_equity": max_equity / 100, "min_bonds": min_bonds / 100, "max_other": max_other / 100,
     "max_crypto": max_crypto / 100, "max_weight_per_asset": max_weight / 100}

# Save the choices for the next pages
st.session_state["assets"] = tickers
st.session_state["constraints"] = c

# ---------- Explain optional requests separately from mandatory prototype limits ----------
riskier = []
if c["max_equity"] > rec_c["max_equity"]:
    riskier.append(f"more equities than recommended ({c['max_equity']:.0%} instead of {rec_c['max_equity']:.0%})")
if c["min_bonds"] < rec_c["min_bonds"]:
    riskier.append(f"fewer bonds than recommended ({c['min_bonds']:.0%} instead of {rec_c['min_bonds']:.0%})")
if c["max_weight_per_asset"] > rec_c["max_weight_per_asset"]:
    riskier.append(f"less diversification ({c['max_weight_per_asset']:.0%} per product instead of "
                   f"{rec_c['max_weight_per_asset']:.0%})")
if c["max_other"] > rec_c["max_other"]:
    riskier.append(f"more gold and commodities than recommended ({c['max_other']:.0%} instead of "
                   f"{rec_c['max_other']:.0%})")
if c["max_crypto"] > rec_c["max_crypto"]:
    riskier.append(f"more crypto than recommended ({c['max_crypto']:.0%} instead of {rec_c['max_crypto']:.0%})")
if riskier:
    st.info("Your optional request relaxes the profile defaults: " + "; ".join(riskier) + ".")
if automatic:
    requested = c
    c = effective_constraints(profile.mandatory_constraints, requested)
    st.dataframe({"Limit": list(c),
                  "Mandatory prototype limit": list(profile.mandatory_constraints.values()),
                  "Optional request": [requested[key] for key in c],
                  "Execution limit": list(c.values())}, hide_index=True)
    if requested != c:
        st.info("The relaxed requests above are not applied. Execution retains every mandatory prototype limit.")
if any(ASSETS[t][1] == "Crypto" for t in tickers) and c["max_crypto"] == 0:
    st.info("You chose crypto products but your maximum in crypto is 0%: they will get no weight.")

problems = allocation_feasibility(tickers, c).problems
show_profile_notice(profile, extra_warnings=problems)
for p in problems:
    st.error(p)
if not problems:
    st.success(f"{len(tickers)} products selected. Allocation limits are numerically feasible; "
               "financial suitability has not been validated.")
else:
    st.info("The portfolio page will try a clearly labelled illustration using mandatory limits and available default products. "
            "Your requested products and preferences will remain saved.")
st.page_link("app_pages/portfolio.py", label="Next: see portfolio or simulation diagnostics", icon="➡️")
