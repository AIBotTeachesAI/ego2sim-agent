"""OpenCV placement check on a rendered top-down image (needs MuJoCo and third_party/mujoco_menagerie)."""
import os

import numpy as np
import pytest

if not os.path.isdir("third_party/mujoco_menagerie/franka_emika_panda"):
    pytest.skip("third_party/mujoco_menagerie not found (see README setup)", allow_module_level=True)

from ego2sim.agent.scene import VerifyScene
from ego2sim.agent.verify import measure

TARGET = np.array([0.59, 0.12])


@pytest.fixture(scope="module")
def scene():
    s = VerifyScene()
    yield s
    s.close()


def test_cup_on_target_is_success(scene):
    scene.set_state(scene.home, TARGET)
    m = measure(scene.render_topdown(), TARGET)
    assert m["success"] and m["error_cm"] < 0.5 and m["workspace_clear"]


def test_known_offset_is_measured_within_3_mm(scene):
    offset = np.array([0.08, -0.06])                       # 8 cm forward, 6 cm to the robot's right
    scene.set_state(scene.home, TARGET + offset)
    m = measure(scene.render_topdown(), TARGET)
    assert not m["success"]
    assert np.allclose(m["error_xy_cm"], offset * 100, atol=0.3)
    assert abs(m["error_cm"] - 10.0) < 0.3
