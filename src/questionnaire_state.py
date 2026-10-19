"""Draft answers live independently of Streamlit widget lifetimes."""
from copy import deepcopy

from src.financial_capacity import POLICY_VERSION


def initialize_draft(state):
    if "questionnaire_answers" not in state:
        state["questionnaire_answers"] = deepcopy(getattr(state.get("profile"), "questionnaire_answers", {}))
    if "capacity_answers" not in state:
        state["capacity_answers"] = deepcopy(getattr(state.get("profile"), "capacity_answers", {}))


def save_widget(state, store, answer_key, widget_key):
    state[store][answer_key] = deepcopy(state[widget_key])


def publish_profile(state, profile, *, migrate=False):
    old = state.get("profile")
    is_migration = (old is not None and getattr(old, "capacity_policy_version", None) != POLICY_VERSION
                    and getattr(profile, "capacity_policy_version", None) == POLICY_VERSION)
    if is_migration and not migrate:
        raise ValueError("Review and explicitly apply the saved profile's policy migration first.")
    profile.migration_history = deepcopy(getattr(old, "migration_history", []))
    if is_migration:
        snapshot = old.policy_snapshot()
        snapshot["saved_optional_constraints"] = deepcopy(state.get("constraints"))
        snapshot["saved_asset_choices"] = deepcopy(state.get("assets"))
        profile.migration_history.append(snapshot)
        state["previous_profile"] = deepcopy(old)
    state["profile"] = profile
    state.pop("capacity_migration_candidate", None)
    for key in list(state):
        if key in ("assets", "constraints") or key.startswith("pick_"):
            del state[key]
