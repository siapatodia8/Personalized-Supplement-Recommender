"""
Shared setup for the entity extractor unit tests.

- Puts the project root on sys.path so `src.*` imports work from any cwd.
- Loads .env so the live tests can reach the Anthropic API.
- Registers the `llm` marker. Live tests call the real model and are skipped
  when ANTHROPIC_API_KEY is missing. Run only the offline tests with:
      pytest unit_tests/entity_extractor -m "not llm"
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # unit_tests/entity_extractor -> project root
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")


def pytest_configure(config):
    config.addinivalue_line("markers", "llm: calls the real Anthropic API (slow, costs tokens)")


class FakeClient:
    """Stands in for anthropic.Anthropic: returns a fixed reply or raises."""

    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(content=[SimpleNamespace(text=self.reply)])


@pytest.fixture
def fake_client():
    """Factory: fake_client(reply="...") or fake_client(error=Exception())."""
    return FakeClient


@pytest.fixture(scope="session")
def live_client():
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    from anthropic import Anthropic

    return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
