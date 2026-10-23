from types import SimpleNamespace

from grounded.model import ClaudeModel, Usage
from grounded.tools import default_tools


class CapturingClient:
    def __init__(self):
        self.calls = []
        create = lambda **kw: self.calls.append(kw) or SimpleNamespace(content=[], stop_reason="end_turn")
        self.messages = SimpleNamespace(create=create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


def test_claude_request_shape():
    client = CapturingClient()
    model = ClaudeModel(effort="low", client=client)
    model.create(system="s", messages=[{"role": "user", "content": "q"}], tools=default_tools().to_api())
    kw = client.calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["output_config"] == {"effort": "low"}
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["cache_control"] == {"type": "ephemeral"}
    assert "thinking" not in kw and "temperature" not in kw and "tool_choice" not in kw
    for tool in kw["tools"]:
        assert tool["strict"] is True and tool["input_schema"]["additionalProperties"] is False


def test_cost_estimate_uses_published_rates():
    u = Usage(input_tokens=1_000_000, output_tokens=100_000, cache_read_tokens=1_000_000)
    assert round(u.cost("claude-opus-5-5"), 4) == round(4.00 + 2.00 + 0.20, 4)
    assert u.cost("some-unknown-model") is None
