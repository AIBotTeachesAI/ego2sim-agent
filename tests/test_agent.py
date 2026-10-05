"""Agent decisions with a mocked Bedrock client (no AWS account needed) and the code-level safety gate."""
import numpy as np

from ego2sim.agent.bedrock_agent import BedrockAgent, describe_error
from ego2sim.agent.loop import MAX_MOVE_M, apply_safety, trustworthy

MEAS = {"detected_m": [0.65, 0.20], "target_m": [0.59, 0.12], "error_xy_cm": [6.0, 8.0], "error_cm": 10.0,
        "error_dir": "+x, +y", "cup_conf": 0.95, "success": False, "workspace_clear": True, "obstructions": []}


class FakeBedrock:
    """Stands in for boto3's bedrock-runtime client."""

    def __init__(self, content=None, error=None):
        self.content, self.error, self.calls = content, error, []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return {"output": {"message": {"content": self.content}}, "usage": {"inputTokens": 100, "outputTokens": 20}}


def agent(**kw):
    return BedrockAgent(model_id="test-model", region="us-west-2", client=FakeBedrock(**kw))


def test_retry_place_is_parsed():
    a = agent(content=[{"toolUse": {"name": "retry_place", "input": {"dx_cm": -6, "dy_cm": -8, "reason": "off target"}}}])
    d = a.decide(MEAS, [])
    assert d["tool"] == "retry_place"
    assert d["args"] == {"dx_cm": -6.0, "dy_cm": -8.0}
    assert d["reason"] == "off target"


def test_accept_and_ask_human_carry_no_move():
    for tool in ("accept", "ask_human"):
        d = agent(content=[{"toolUse": {"name": tool, "input": {"reason": "r"}}}]).decide(MEAS, [])
        assert d["tool"] == tool and d["args"] == {}


def test_text_only_reply_falls_back_to_human():
    d = agent(content=[{"text": "I think it is fine."}]).decide(MEAS, [])
    assert d["tool"] == "ask_human"


def test_bedrock_error_falls_back_to_human():
    d = agent(error=RuntimeError("throttled")).decide(MEAS, [])
    assert d["tool"] == "ask_human" and d["args"] == {}


def test_observation_sent_to_model_has_the_opencv_measurement():
    a = agent(content=[{"toolUse": {"name": "accept", "input": {"reason": "ok"}}}])
    a.decide(MEAS, [])
    sent = a.client.calls[0]["messages"][0]["content"][0]["text"]
    assert '"error_cm": 10.0' in sent and '"workspace_clear": true' in sent


def test_error_is_described_in_plain_words():
    words = describe_error([6.0, 8.0])
    assert "forward" in words and "left" in words
    assert describe_error([0.0, 0.0]) == "on target"
    assert "unknown" in describe_error(None)


def test_safety_gate_clamps_large_moves():
    new, move, clamped, inside = apply_safety([0.50, 0.00], {"dx_cm": 40.0, "dy_cm": 0.0})
    assert clamped and np.isclose(np.linalg.norm(move), MAX_MOVE_M)
    assert np.allclose(new, [0.50 + MAX_MOVE_M, 0.0]) and inside


def test_safety_gate_rejects_destinations_outside_workspace():
    _, _, clamped, inside = apply_safety([0.70, 0.20], {"dx_cm": 10.0, "dy_cm": 10.0})
    assert not clamped and not inside


def test_untrusted_scenes_are_flagged():
    assert trustworthy(MEAS)
    assert not trustworthy({**MEAS, "cup_conf": 0.35})
    assert not trustworthy({**MEAS, "workspace_clear": False})
    assert not trustworthy({**MEAS, "detected_m": None})
