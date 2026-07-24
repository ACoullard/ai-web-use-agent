import pytest

from webagent.providers import (
    ProviderConfigError,
    ThinkingLevel,
    check_model_config,
    resolve_thinking,
)


def test_check_model_config_passes_with_key_set():
    check_model_config("openai:gpt-4o", env={"OPENAI_API_KEY": "sk-test"})


def test_check_model_config_unknown_provider_raises():
    with pytest.raises(ProviderConfigError, match="Unsupported or missing provider"):
        check_model_config("google:gemini-2.0", env={"GOOGLE_API_KEY": "x"})


def test_check_model_config_missing_prefix_raises():
    with pytest.raises(ProviderConfigError, match="Unsupported or missing provider"):
        check_model_config("claude-sonnet-5", env={"ANTHROPIC_API_KEY": "x"})


def test_check_model_config_missing_key_raises():
    with pytest.raises(ProviderConfigError, match="OPENAI_API_KEY is not set"):
        check_model_config("openai:gpt-4o", env={})


def test_check_model_config_reads_os_environ_by_default(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ProviderConfigError, match="ANTHROPIC_API_KEY is not set"):
        check_model_config("anthropic:claude-sonnet-5")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    check_model_config("anthropic:claude-sonnet-5")


@pytest.mark.parametrize(
    "level,expected",
    [
        (ThinkingLevel.OFF, False),
        (ThinkingLevel.MINIMAL, "minimal"),
        (ThinkingLevel.MEDIUM, "medium"),
        (ThinkingLevel.XHIGH, "xhigh"),
    ],
)
def test_resolve_thinking(level, expected):
    result = resolve_thinking(level)
    assert result == expected
    assert type(result) is type(expected)
