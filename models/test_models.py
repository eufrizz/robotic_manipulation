import mujoco
from pathlib import Path
DIR = Path(__file__).parent.resolve()   # note: absolute path

file = DIR/"lite6_viz.xml"
model = mujoco.MjModel.from_xml_path(str(file))
print(f"Loaded {file}")

# file = DIR/"cube_pickup.xml"
# model = mujoco.MjModel.from_xml_path(str(file))
# print(f"Loaded {file}")