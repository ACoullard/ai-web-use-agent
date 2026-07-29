import enum
import os
from collections.abc import Mapping
from typing import Any

# Provider prefix (the part before ':' in a "<provider>:<model>" identifier) mapped to
# the environment variable holding its API key. pydantic-ai reads the key from the env
# itself; this table just lets us fail early with a friendly message.
SUPPORTED_PROVIDERS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


class ProviderConfigError(Exception):
    """Raised when a model identifier names an unsupported provider or its key is unset."""


class ThinkingLevel(str, enum.Enum):
    """Named reasoning/thinking effort levels (see `resolve_thinking` for the mapping)."""

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    OFF = "off"


DEFAULT_THINKING: ThinkingLevel = ThinkingLevel.MEDIUM
# DEFAULT_MODEL = "anthropic:claude-sonnet-5"
DEFAULT_MODEL = "openai:gpt-5.6-terra"

# How much reasoning-summary detail to ask OpenAI reasoning models for: "auto" is the most
# detailed summary a given model allows.
OPENAI_REASONING_SUMMARY = "auto"


def resolve_thinking(level: ThinkingLevel) -> str | bool:
    """Map a `ThinkingLevel` to Pydantic AI's unified `thinking` setting (`off` -> False)."""
    return False if level is ThinkingLevel.OFF else level.value


def build_model_settings(model: str, thinking: str | bool) -> dict[str, Any]:
    """Model settings for an LLM call: the thinking effort, plus whatever it takes to get the
    model's reasoning back in the response so traces can record it.

    Anthropic returns thinking text whenever thinking is on, so `thinking` alone is enough
    there. OpenAI reasoning models never return raw reasoning - it comes back encrypted
    unless a summary is requested, so we always request one. With thinking off there is no
    reasoning to summarize, so the setting is omitted rather than sent alongside effort
    `none`.
    """
    settings: dict[str, Any] = {"thinking": thinking}
    if model.split(":", 1)[0] == "openai" and thinking is not False:
        settings["openai_reasoning_summary"] = OPENAI_REASONING_SUMMARY
    return settings


def check_model_config(model: str, *, env: Mapping[str, str] | None = None) -> None:
    """Validate that `model` uses a supported provider and its API key is available.

    `model` is a Pydantic AI identifier of the form "<provider>:<model>", e.g.
    "anthropic:claude-sonnet-5" or "openai:gpt-4o". Raises ProviderConfigError with an
    actionable message if the provider prefix is unknown or the required key env var is
    not set.
    """
    env = os.environ if env is None else env

    provider = model.split(":", 1)[0] if ":" in model else ""
    if provider not in SUPPORTED_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
        raise ProviderConfigError(
            f"Unsupported or missing provider in model {model!r}. "
            f"Use the form '<provider>:<model>' with one of: {supported}."
        )

    env_var = SUPPORTED_PROVIDERS[provider]
    if not env.get(env_var):
        raise ProviderConfigError(
            f"{env_var} is not set, which is required to use {provider} models. "
            f"Export it and retry (e.g. `export {env_var}=...`)."
        )
