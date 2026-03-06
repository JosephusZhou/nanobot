"""Task-based model routing for provider/model fallbacks."""

from __future__ import annotations

import re
from dataclasses import dataclass

from nanobot.providers.base import LLMProvider

_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```", re.M)
_CODE_SIGNAL_RE = re.compile(
    r"\b("
    r"python|javascript|typescript|java|kotlin|go|rust|swift|sql|regex|"
    r"api|sdk|framework|library|debug|bug|refactor|compile|build|test|"
    r"docker|k8s|kubernetes|git|repo|stack trace|exception|traceback"
    r")\b",
    re.I,
)

# Includes Chinese + English tokens for development/code/technical doc tasks.
_CODING_KEYWORDS = (
    "代码", "编程", "开发", "调试", "报错", "错误栈", "重构", "修复", "单测", "测试用例",
    "技术文档", "开发文档", "官方文档", "api 文档", "sdk 文档", "框架文档",
    "coding", "programming", "developer", "software", "implementation",
)


@dataclass(frozen=True)
class ModelRoute:
    """A provider/model pair for a task category."""

    provider: LLMProvider
    model: str
    keywords: tuple[str, ...] = ()


def _normalize(text: str | None) -> str:
    return (text or "").strip().lower()


def is_coding_task(text: str | None) -> bool:
    """Heuristically detect coding or dev-knowledge tasks."""
    content = _normalize(text)
    if not content:
        return False
    if _CODE_FENCE_RE.search(content):
        return True
    if _CODE_SIGNAL_RE.search(content):
        return True
    return any(token in content for token in _CODING_KEYWORDS)


def select_route(task_text: str | None, routes: dict[str, ModelRoute]) -> str | None:
    """Return matched route name, or None when default model should be used."""
    if not routes:
        return None

    text = _normalize(task_text)
    if not text:
        return None

    # User-defined keyword routes have highest priority.
    for route_name, route in routes.items():
        if any(kw and kw.lower() in text for kw in route.keywords):
            return route_name

    # Built-in coding classifier for a common route name.
    if "coding" in routes and is_coding_task(text):
        return "coding"

    return None
