"""Claude on Amazon Bedrock as the decision step (flow step 6).

Input: the OpenCV measurement (numbers only). Output: exactly one tool call, via Bedrock tool use:
  retry_place(dx_cm, dy_cm, reason) | accept(reason) | ask_human(reason)
The loop's safety gate still decides what actually executes.
"""
import json
import os
import time

import boto3

from .loop import MAX_CORRECTIONS, MAX_MOVE_M, TOL_CM, WORKSPACE

SYSTEM = f"""You are the decision module of a robot arm that places a cup on a table.
After every placement, an OpenCV vision check measures where the cup is relative to the target.
You receive that measurement as JSON and must call exactly one tool.

Tools:
- accept: the cup is placed well enough. Use only if the error is below {TOL_CM} cm AND the measurement is trustworthy.
- retry_place: move the cup by (dx_cm, dy_cm). To cancel a measured error (ex, ey) the move is (-ex, -ey).
  Axes: x = forward (away from the robot), y = robot's left. Units: cm.
- ask_human: stop and ask the human operator for approval.

Rules:
- If the measurement is trustworthy, the workspace is clear and error_cm is below the tolerance, you MUST call accept.
  Do not fine-tune: each re-place adds about 0.5 cm of random placing error, so corrections smaller than
  the tolerance only chase noise and can push the cup out of tolerance.
- A measurement is trustworthy only if cup_found is true and cup_conf >= 0.5. Otherwise ask_human.
- If workspace_clear is false, an unexpected object (possibly a person's hand) is on the table:
  never move the cup; ask_human.
- At most {MAX_CORRECTIONS} corrections per task (see corrections_used). If none are left, ask_human.
- If the previous correction did not cut the error at least in half (see last_correction_error_ratio),
  something is wrong (for example the gripper is slipping): ask_human instead of repeating it.
- The destination (detected_cup_m + move) must stay inside the workspace {WORKSPACE} (meters).
  If the correction would leave it, ask_human.
- The safety layer caps any move at {MAX_MOVE_M * 100:.0f} cm and rejects unsafe destinations.
- Describe directions using error_in_words (it is already translated from the axis signs).
- Give a one-sentence reason."""

TOOLS = {"tools": [
    {"toolSpec": {
        "name": "retry_place",
        "description": "Pick the cup up and place it again, moved by (dx_cm, dy_cm) from where OpenCV detected it.",
        "inputSchema": {"json": {"type": "object", "properties": {
            "dx_cm": {"type": "number", "description": "move along x (forward), cm"},
            "dy_cm": {"type": "number", "description": "move along y (robot's left), cm"},
            "reason": {"type": "string"}}, "required": ["dx_cm", "dy_cm", "reason"]}}}},
    {"toolSpec": {
        "name": "accept",
        "description": "The cup is on target; finish the task.",
        "inputSchema": {"json": {"type": "object", "properties": {"reason": {"type": "string"}},
                                 "required": ["reason"]}}}},
    {"toolSpec": {
        "name": "ask_human",
        "description": "Stop and ask the human operator whether to try again.",
        "inputSchema": {"json": {"type": "object", "properties": {"reason": {"type": "string"}},
                                 "required": ["reason"]}}}},
], "toolChoice": {"any": {}}}


def _load_env(path=".env"):
    env = {}
    if os.path.exists(path):
        for line in open(path):
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                env[k] = v
    return env


def describe_error(err_xy_cm):
    """Plain-words error, so the model never has to translate axis signs."""
    if err_xy_cm is None:
        return "unknown (cup not found)"
    ex, ey = err_xy_cm
    parts = []
    if abs(ex) >= 0.1:
        parts.append(f"{abs(ex):.1f} cm too far {'forward (away from the robot)' if ex > 0 else 'back (toward the robot)'}")
    if abs(ey) >= 0.1:
        parts.append(f"{abs(ey):.1f} cm too far to the robot's {'left' if ey > 0 else 'right'}")
    return ", ".join(parts) if parts else "on target"


def _last_ratio(meas, history):
    """error now / error before the last executed correction (None if no correction yet)."""
    if not history or not history[-1].get("executed") or meas["error_cm"] is None:
        return None
    before = history[-1]["opencv"]["error_cm"]
    return round(meas["error_cm"] / before, 2) if before else None


def build_observation(meas, history):
    return {
        "error_in_words": describe_error(meas["error_xy_cm"]),
        "check": len(history) + 1,
        "cup_found": meas["detected_m"] is not None,
        "cup_conf": meas["cup_conf"],
        "detected_cup_m": meas["detected_m"],
        "target_m": meas.get("target_m"),
        "error_xy_cm": meas["error_xy_cm"],
        "error_cm": meas["error_cm"],
        "error_dir": meas["error_dir"],
        "workspace_clear": meas.get("workspace_clear", True),
        "unexpected_objects_in_workspace": len(meas.get("obstructions", [])),
        "tolerance_cm": TOL_CM,
        "corrections_used": sum(1 for s in history if s.get("executed")),
        "last_correction_error_ratio": _last_ratio(meas, history),
        "max_corrections": MAX_CORRECTIONS,
        "workspace_m": WORKSPACE,
        "history": [{"check": s["check"], "error_cm": s["opencv"]["error_cm"], "cup_conf": s["opencv"]["cup_conf"],
                     "tool": s["decision"].get("tool"), "move_cm": s.get("safety", {}).get("applied_move_cm"),
                     "executed": s["executed"]} for s in history],
    }


class BedrockAgent:
    def __init__(self, model_id=None, region=None, client=None):
        env = _load_env()
        self.model_id = model_id or env.get("BEDROCK_MODEL_ID")
        region = region or env.get("AWS_REGION", "us-west-2")
        if not self.model_id:
            raise ValueError("BEDROCK_MODEL_ID missing from .env")
        self.client = client or boto3.client("bedrock-runtime", region_name=region)

    def decide(self, meas, history):
        obs = build_observation(meas, history)
        t0 = time.time()
        info = {"model": self.model_id}
        try:
            resp = self.client.converse(
                modelId=self.model_id,
                system=[{"text": SYSTEM}],
                messages=[{"role": "user", "content": [{"text": json.dumps(obs)}]}],
                toolConfig=TOOLS,
                inferenceConfig={"maxTokens": 300, "temperature": 0},
            )
            info.update(latency_s=round(time.time() - t0, 3),
                        input_tokens=resp["usage"]["inputTokens"], output_tokens=resp["usage"]["outputTokens"])
            for block in resp["output"]["message"]["content"]:
                if "toolUse" in block:
                    name, args = block["toolUse"]["name"], dict(block["toolUse"].get("input", {}))
                    reason = str(args.pop("reason", ""))
                    if name == "retry_place":
                        args = {"dx_cm": float(args.get("dx_cm", 0.0)), "dy_cm": float(args.get("dy_cm", 0.0))}
                    else:
                        args = {}
                    return {"tool": name, "args": args, "reason": reason, "llm": info}
            reason = "LLM returned no tool call"
        except Exception as e:                       # network, throttling, access: never act on a failed call
            info["latency_s"] = round(time.time() - t0, 3)
            reason = f"LLM call failed: {type(e).__name__}"
            info["error"] = str(e)[:300]
        return {"tool": "ask_human", "args": {}, "reason": f"{reason}; falling back to human", "llm": info}
