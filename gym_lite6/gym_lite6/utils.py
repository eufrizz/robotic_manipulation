from scipy.linalg import logm, expm
import numpy as np
import mujoco

def s(t, end_time):
  """
  Calculate a third order polynomial time scaling s
  s = a0 + a1*t + a2*t^2 + a3*t^3
  where
  s = 0 at t=0 and s=end_time at t=end_time
  sdot is 0 at t=0 and t=end_time

  Vectorised implementation allows caluclation of multiple t values simultaneously

  Args:
    t: a time or array of times for which to calculate s
    end_time: the duration of the trajectory, shorter means faster

  Returns:
    s: s value
    s_1: first derivative of s
    s_2: second derivative of s
  """
  t = np.array(t)
  assert(np.all(t >= 0) and np.all(t <= end_time))
  a0 = 0
  a1 = 0
  a2 = 3/end_time**2
  a3 = -2/end_time**3

  # Constants for s, sdot, sddot
  A = np.array([[a0, a1, a2, a3],
                [a1, 2*a2, 3*a3, 0],
                [2*a2, 6*a3, 0, 0]])

  # Form a matrix of
  # [1,     1,     ...]
  # [t0,    t1,    ...]
  # [t0**2, t1**2, ...]
  # [t0**3, t1**3, ...]
  x = np.ones(t.size)
  for i in range(1, 4):
      x = np.vstack((x, t**i))

  assert(x.shape == (4, t.size))

  s = A @ x
  s_0 = s[0, :]
  s_1 = s[1, :]
  s_2 = s[2, :]
  return s_0, s_1, s_2

def get_tf_matrix(pos, quat):
  pos = np.reshape(np.atleast_2d(pos), (3, 1))
  R = np.zeros(9)
  mujoco.mju_quat2Mat(R, quat)
  R = np.reshape(R, (3, 3))
  T = np.hstack((R, pos))
  T = np.vstack((T, [0, 0, 0, 1]))
  return T

def tf_matrix_to_pose_quat(T):
  R = T[:3, :3].reshape((9, 1))
  pos = list(T[:3, 3])
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, R)
  return pos, quat

def invert_tf_matrix(T):
  R = T[:3, :3]
  t = T[:3, 3]
  T_inv = np.eye(4)
  T_inv[:3, :3] = R.T
  T_inv[:3, 3] = -R.T @ t
  return T_inv

def screw_interp(T_start, T_end, t, end_time):
  """ 
  Screw interpolation calculates a constant twist to get from the start pose to the end pose in the given time.
  This function calculates the pose at a given time t along the trajectory, using our time scaling s(t) from before.
  Args:
    T_start: start pose, as a 4x4 transform matrix
    T_end: end pose, as a 4x4 transform matrix
    t: the time at which to s
    max vel = 3/(2*end_time)
  """
  assert(t >= 0 and t <= end_time)
  return T_start @ expm(logm(invert_tf_matrix(T_start)@T_end)*s(t, end_time)[0])


def plot_dict_of_arrays(ep_dict, x_ax, keys=None, title_prefix="", sharey=True):
  import matplotlib.pyplot as plt

  # if x_ax not in ep_dict:
  #   raise KeyError("x_ax must be a member of ep_dict")
  if keys is None:
    keys = [key for key in ep_dict.keys() if key != x_ax]
    
  
  for key in keys:
    if hasattr(ep_dict[key][0], '__iter__') and ep_dict[key][0].shape:
      len_state = len(ep_dict[key][0])
    else:
      len_state = 1
    ncols = len_state; nrows= int(np.ceil(len_state/ncols))
    plt_data = np.array(ep_dict[key])
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, sharex=True, sharey=sharey, figsize=(ncols*2+0.5, nrows*2+1), constrained_layout=True)

    if len_state > 1:
      for i in range(len_state):
        ax = axs.flatten()[i]
        ax.plot(ep_dict[x_ax], plt_data[:, i])
        ax.set_title(i)
    else:
      ax = axs
      ax.plot(ep_dict[x_ax], plt_data)

    # fig.add_subplot(111, frameon=False)
    plt.suptitle(f"{key}")
    # fig.supylabel("Joint angle")
    fig.supxlabel(x_ax)