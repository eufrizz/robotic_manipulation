import mujoco
from pathlib import Path
DIR = Path(__file__).parent.resolve()   # note: absolute path
model = mujoco.MjModel.from_xml_path(str(DIR/"cube_pickup.xml"))