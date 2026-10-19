"""Streamlit interaction tests run offline against the profile page."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np

from streamlit.testing.v1 import AppTest

from test_questionnaire import CAPACITY, completed_answers
from src.client_profile import ClientProfile
from src.financial_capacity import POLICY_VERSION
from src.data import ASSETS
from src.portfolios import build_portfolios, estimate


PAGE = Path(__file__).resolve().parents[1] / "app.py"
FINANCIAL = dict(age="30-45", income="Very stable", emergency_savings="3-6 months", commitments="Manageable")


class QuestionnaireUITests(unittest.TestCase):
    def setUp(self):
        # Exercise real st.navigation/page links; stub only external market data.
        market_data = pd.DataFrame({"SPY": [100.0]}, index=pd.to_datetime(["2026-10-15"]))
        patcher = patch("src.data.download_prices", return_value=market_data)
        patcher.start()
        self.addCleanup(patcher.stop)

    def page(self, answers=None):
        app = AppTest.from_file(str(PAGE), default_timeout=15)
        if answers is not None:
            app.session_state["questionnaire_answers"] = deepcopy(answers)
        app.run()
        self.assertEqual(len(app.exception), 0)
        return app

    def section(self, app, section):
        app.selectbox(key="questionnaire_section_selector").select(section).run()
        self.assertEqual(len(app.exception), 0)

    def test_initial_render_and_next_navigation(self):
        app = self.page()
        self.assertEqual(sum(item.key.startswith("questionnaire_widget_QA") for item in app.radio), 2)
        self.assertFalse(any("Holt" in str(element.value) for element in app.markdown))
        app.radio(key="questionnaire_widget_QA1").set_value("a").run()
        app.radio(key="questionnaire_widget_QA2").set_value("c").run()
        next(button for button in app.button if button.label == "Next section").click().run()
        self.assertEqual(app.session_state["questionnaire_section"], "B")
        self.section(app, "A")
        self.assertEqual(app.radio(key="questionnaire_widget_QA1").value, "a")

    def test_hidden_answers_restore_and_scores_exclude_them(self):
        app = self.page(completed_answers(QC1="c", QC4=["h"]))
        self.section(app, "C")
        self.assertEqual(app.multiselect(key="questionnaire_widget_QC4").value, ["h"])
        app.radio(key="questionnaire_widget_QC1").set_value("a").run()
        self.assertFalse(any(item.key == "questionnaire_widget_QC4" for item in app.multiselect))
        self.assertEqual(app.session_state["questionnaire_answers"]["QC4"], ["h"])
        self.section(app, "A")
        self.section(app, "C")
        app.radio(key="questionnaire_widget_QC1").set_value("c").run()
        self.assertEqual(app.multiselect(key="questionnaire_widget_QC4").value, ["h"])

    def test_incomplete_submission_does_not_publish(self):
        app = self.page()
        self.section(app, "Results")
        next(button for button in app.button if button.label == "Compute my profile").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertGreater(len(app.error), 0)
        self.assertNotIn("profile", app.session_state)

    def test_capacity_automatic_gamma_and_questionnaire_survive_navigation_and_submit(self):
        app = self.page(completed_answers())
        self.section(app, "Capacity")
        for name, value in FINANCIAL.items():
            app.selectbox(key=f"questionnaire_capacity_{name}").set_value(value).run()
        self.assertFalse(any(item.key == "questionnaire_gamma_input" for item in app.number_input))
        self.section(app, "A")
        self.section(app, "Capacity")
        self.assertEqual(len(app.number_input), 0)
        self.assertEqual(app.selectbox(key="questionnaire_capacity_income").value, CAPACITY["income"])
        self.section(app, "Results")
        next(button for button in app.button if button.label == "Compute my profile").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        profile = app.session_state["profile"]
        from src.gamma_policy import DEFAULT_RESEARCH_MAPPING
        self.assertAlmostEqual(profile.gamma_willingness, DEFAULT_RESEARCH_MAPPING.willingness_gamma(profile.rt))
        self.assertEqual(profile.gamma, max(profile.gamma_willingness, profile.gamma_capacity))
        self.assertEqual(profile.rt, 56.25)
        self.assertTrue(profile.gamma_provisional)
        self.assertTrue(profile.gamma_automatic)
        self.assertEqual(profile.capacity_policy_version, POLICY_VERSION)
        self.assertEqual(profile.capacity_score, 6)
        self.assertEqual(profile.raw_horizon_answers, {"QD1": "d", "QD2": "d"})
        self.assertTrue(any("academic" in str(element.value).lower() for element in app.caption))
        self.assertEqual(len(app.warning), 1)
        self.assertTrue(any("withdrawal" in str(element.value).lower() for element in app.warning))

    def test_views_persist_and_hidden_cv_disappears(self):
        app = self.page(completed_answers(QA2="g", QH4="b"))
        self.section(app, "H")
        next(button for button in app.button if button.label == "Add a quantitative view").click().run()
        app.text_input(key="questionnaire_view_0_asset").set_value("SPY").run()
        app.number_input(key="questionnaire_view_0_value").set_value(7.5).run()
        app.radio(key="questionnaire_view_0_confidence").set_value("d").run()
        self.section(app, "A")
        self.section(app, "H")
        self.assertEqual(app.text_input(key="questionnaire_view_0_asset").value, "SPY")
        self.assertEqual(app.radio(key="questionnaire_view_0_confidence").value, "d")
        app.radio(key="questionnaire_widget_QH4").set_value("a").run()
        self.assertEqual(len(app.number_input), 0)
        self.assertEqual(app.session_state["questionnaire_answers"]["QH5"][0]["value"], 7.5)

    def test_explicit_none_and_advanced_opt_in(self):
        app = self.page(completed_answers(QB1="a", QB2=["none"]))
        self.section(app, "B")
        self.assertFalse(any(item.key == "questionnaire_widget_QB3" for item in app.radio))
        self.assertEqual(app.multiselect(key="questionnaire_widget_QB2").value, ["none"])
        app.checkbox(key="questionnaire_widget_advanced_enabled").check().run()
        self.section(app, "F")
        self.assertTrue(any(item.key == "questionnaire_widget_QF5" for item in app.radio))
        app.checkbox(key="questionnaire_widget_advanced_enabled").uncheck().run()
        self.assertFalse(any(item.key == "questionnaire_widget_QF5" for item in app.radio))

    def test_draft_survives_switching_to_another_app_page(self):
        app = self.page(completed_answers())
        self.section(app, "E")
        app.radio(key="questionnaire_widget_QE1").set_value("e").run()
        app.switch_page("app_pages/how_it_works.py").run()
        self.assertEqual(len(app.exception), 0)
        app.switch_page("app_pages/profile.py").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["questionnaire_answers"]["QE1"], "e")
        self.assertEqual(app.session_state["questionnaire_section"], "E")
        self.assertEqual(app.radio(key="questionnaire_widget_QE1").value, "e")

    def test_optional_relaxations_do_not_bypass_limits_and_risk_parity_shows_no_gamma(self):
        dates = pd.date_range("2020-01-31", periods=80, freq="ME")
        rng = np.random.default_rng(42)
        returns = pd.DataFrame(rng.normal(.004, .025, (80, len(ASSETS))),
                               index=dates, columns=list(ASSETS))
        prices = 100 * (1 + returns).cumprod()
        rf = pd.Series(.001, index=dates)
        profile = ClientProfile.from_questionnaire(completed_answers(), CAPACITY)
        captured = []

        def portfolios(currency, tickers, gamma, c, views):
            captured.append(dict(c))
            mu, cov = estimate(returns[tickers])
            return build_portfolios(mu, cov, .012, gamma, c, views, n_sim=2)

        with patch("src.data.download_prices", return_value=prices), patch(
                "src.app_helpers.get_data", return_value=(returns, rf, prices)), patch(
                "src.app_helpers.get_portfolios", side_effect=portfolios):
            app = self.page(completed_answers())
            app.session_state["profile"] = profile
            app.session_state["capacity_answers"] = deepcopy(CAPACITY)
            app.switch_page("app_pages/choices.py").run()
            self.assertEqual(len(app.exception), 0)
            app.slider(key="pick_max_equity").set_value(100).run()
            app.slider(key="pick_max_weight").set_value(100).run()
            self.assertEqual(app.session_state["constraints"]["max_equity"], 1)
            self.assertTrue(any("not applied" in str(item.value) for item in app.info))
            app.switch_page("app_pages/portfolio.py").run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(captured[-1], profile.mandatory_constraints)
            metric_names = {metric.label for metric in app.metric}
            self.assertTrue({"Expected return", "Volatility", "Equity exposure", "Bond exposure",
                             "Largest holding", "HHI"}.issubset(metric_names))
            app.radio[0].set_value("Risk parity (ERC)").run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("Risk Parity does not use gamma" in str(item.value) for item in app.caption))
            self.assertFalse(any("what if gamma" in str(item.value).lower() for item in app.subheader))

    def test_stale_manual_gamma_draft_is_ignored_by_automatic_submission(self):
        app = self.page(completed_answers())
        app.session_state["capacity_answers"] = deepcopy(FINANCIAL)
        app.session_state["optimization_gamma_draft"] = 99.
        self.section(app, "Results")
        next(button for button in app.button if button.label == "Compute my profile").click().run()
        self.assertEqual(len(app.exception), 0)
        profile = app.session_state["profile"]
        self.assertTrue(profile.gamma_automatic)
        self.assertLess(profile.gamma, 99)


if __name__ == "__main__":
    unittest.main()
