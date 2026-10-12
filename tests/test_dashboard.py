"""Dashboard smoke tests: app boots headlessly, empty states render, no exceptions."""

from __future__ import annotations

from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "src" / "fxagent" / "dashboard" / "app.py"


def test_dashboard_boots_with_empty_state():
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    assert not at.exception, at.exception


def test_dashboard_sections_render():
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    # sidebar radio exists with all sections
    assert len(at.sidebar.radio) >= 1
    options = at.sidebar.radio[0].options
    for expected in ("Overview", "Equity & P&L", "Trades", "Risk & Health"):
        assert expected in options


@pytest.mark.parametrize("section", ["Overview", "Equity & P&L", "Trades", "Risk & Health"])
def test_section_labels_present(section):
    """Section option exists in the sidebar radio (script supports it)."""
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    assert section in at.sidebar.radio[0].options
