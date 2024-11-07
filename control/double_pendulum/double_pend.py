import mujoco
from mujoco import viewer
import numpy as np
import time
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt

# Load the model from the XML file
model = mujoco.MjModel.from_xml_path('2D_pendulum.xml')
data = mujoco.MjData(model)
paused = False

class PendulumController(object):
    def __init__(self) -> None:
        assert np.all(np.equal(model.body(1).ipos, np.array([0., 0., 0.])))
        self.m = model.body(1).mass[0]
        self.I = model.body(1).inertia[1]
        self.l = model.geom(0).size[1]
        self.g = -model.opt.gravity[2]

        self.stable_energy = self.m*self.g*self.l
        print(self.m, self.I, self.g, self.l, self.stable_energy)

        self.kin = []
        self.pot = []
        self.t = []
        self.state = 0

    def control_callback(self):
        global model, data
        theta = data.qpos[0]
        omega = data.qvel[0]
        if abs(abs(theta)-np.pi) < 0.1:
            if self.state != 0:
                self.state = 0
                print("Stabilise")
            u = - 5* omega
        else:
            if self.state != 1:
                self.state = 1
                print("Swing")
            kinetic = self.I * omega**2 # When plotted, kinetic was half of potential
            potential  = 0.5*self.m * self.g * self.l*(1- np.cos(theta))
            # self.kin.append(kinetic); self.pot.append(potential); self.t.append(data.time)
            curr_energy = kinetic + potential
            energy_err = self.stable_energy - curr_energy + 0.1
            # print(curr_energy, energy_err)
            u = 5* omega * energy_err 

        u += np.random.normal(0, 0.5)
        data.ctrl = u
    


def key_callback(keycode):
  if chr(keycode) == ' ':
    global paused
    paused = not paused


if __name__ == "__main__":

    pc = PendulumController()
    with viewer.launch_passive(model, data) as viewer:
        # Reset the simulation
        mujoco.mj_resetData(model, data)
        data.qpos[0] = np.random.uniform(-np.pi, np.pi)  # Initial angle of first joint
        # data.qpos[1] = 0.0 
        step_start = time.time()

        while viewer.is_running():# and data.time < 2:
            step_start = time.time()
            
            if not paused:
                pc.control_callback()
                # Step the simulation
                mujoco.mj_step(model, data)
                
                # Update the viewer
                viewer.sync()

                # Get joint positions and velocities
                joint_pos = data.qpos.copy()
                joint_vel = data.qvel.copy()
                
                # Print state (optional)
                # print(f"Joint Positions: {joint_pos}")
                # print(f"Joint Velocities: {joint_vel}")
        
            # Rudimentary time keeping, will drift relative to wall clock.
            # Tends to run at 0.0025s per timestep, instead of 0.002
            time_until_next_step = model.opt.timestep - (time.time() - step_start) - 0.0003
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    
    # plt.plot(pc.t, pc.kin, 'b', label="kinetic")
    # plt.plot(pc.t, pc.pot, 'r', label="potential")
    # plt.plot(pc.t, np.array(pc.pot) + np.array(pc.kin), 'y')
    # plt.legend()
    # plt.savefig("plot.png")