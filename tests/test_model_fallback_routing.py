from __future__ import annotations

from nanobot.agent.model_routing import ModelRoute, select_route
from nanobot.config.loader import _migrate_config
from nanobot.config.schema import Config
from nanobot.providers.base import LLMProvider, LLMResponse


class _DummyProvider(LLMProvider):
    async def chat(self, **kwargs):  # type: ignore[override]
        return LLMResponse(content="ok")

    def get_default_model(self) -> str:
        return "dummy/model"


def test_select_route_uses_custom_keywords_first() -> None:
    routes = {
        "research": ModelRoute(provider=_DummyProvider(), model="m1", keywords=("literature",)),
        "coding": ModelRoute(provider=_DummyProvider(), model="m2"),
    }
    assert select_route("Please do a literature review and write code", routes) == "research"


def test_select_route_detects_coding_tasks() -> None:
    routes = {
        "coding": ModelRoute(provider=_DummyProvider(), model="m2"),
    }
    assert select_route("帮我修复这个 Python 报错并给出代码", routes) == "coding"


def test_model_fallback_config_accepts_camel_case() -> None:
    cfg = Config.model_validate({
        "agents": {
            "defaults": {
                "modelFallback": {
                    "coding": {
                        "provider": "openrouter",
                        "model": "openai/gpt-5",
                        "keywords": ["coding", "debug"],
                    }
                }
            }
        }
    })
    coding = cfg.agents.defaults.model_fallback["coding"]
    assert coding.provider == "openrouter"
    assert coding.model == "openai/gpt-5"
    assert coding.keywords == ["coding", "debug"]


def test_migrate_config_supports_kebab_case_model_fallback() -> None:
    data = {
        "agents": {
            "defaults": {
                "model-fallback": {
                    "coding": {"provider": "openrouter", "model": "openai/gpt-5"}
                }
            }
        }
    }
    migrated = _migrate_config(data)
    defaults = migrated["agents"]["defaults"]
    assert "modelFallback" in defaults
    assert "model-fallback" not in defaults
