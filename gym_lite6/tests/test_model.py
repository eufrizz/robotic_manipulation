"""
In the event of: AttributeError: 'Renderer' object has no attribute '_mjr_context', set the environment variable `export MUJOCO_GL=egl` before running
Pass -s to pytest to show prints/stdout
"""

import mujoco
from pathlib import Path
import pytest

MODEL_DIR = (Path(__file__).parent.parent/"models").resolve()   # note: absolute path

@pytest.fixture
def model():
    file = MODEL_DIR/"lite6_gripper.xml"
    model = mujoco.MjModel.from_xml_path(str(file))
    return model

@pytest.fixture
def data(model):
    data = mujoco.MjData(model)
    return data

@pytest.fixture
def renderer(model):
    renderer = mujoco.Renderer(model, height=224, width=224)
    return renderer

def test_load_model(model, data, renderer):
    print(model)
    print(data)

# def test_pos_actuators(model):
#     print([model.actuator(x) for x in range(model.nu)])
#     # print(model.actuator('gripper'))

