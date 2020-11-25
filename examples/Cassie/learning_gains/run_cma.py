import cma  # https://github.com/CMA-ES/pycma
import subprocess
import time
import numpy as np
from scipy.interpolate import CubicSpline
import lcm
import sys
import yaml
import matplotlib.pyplot as plt

import pydrake
from pydrake.trajectories import PiecewisePolynomial
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.systems.framework import DiagramBuilder

import pydairlib.multibody
import pydairlib.lcm_trajectory
from pydairlib.common import FindResourceOrThrow
from pydairlib.multibody.kinematic import DistanceEvaluator
from pydairlib.cassie.cassie_utils import *
import pydairlib.analysis_scripts.process_lcm_log as process_lcm_log


def obj_func(x):
  # Unpack args
  # plant = args[0]
  # pos_map = args[1]
  # vel_map = args[2]
  # act_map = args[3]

  # Build MBP
  pydrake.common.set_log_level("err")
  builder = DiagramBuilder()
  plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
  pydairlib.cassie.cassie_utils.addCassieMultibody(plant, scene_graph, True,
    "examples/Cassie/urdf/cassie_v2.urdf", False, False)
  plant.Finalize()

  pos_map = pydairlib.multibody.makeNameToPositionsMap(plant)
  vel_map = pydairlib.multibody.makeNameToVelocitiesMap(plant)
  act_map = pydairlib.multibody.makeNameToActuatorsMap(plant)

  log_path = 'testlog'
  logger_cmd = ['lcm-logger',
                '-f',
                '%s' % log_path,
                ]
  simulation_cmd = ['bazel-bin/examples/Cassie/run_sim_and_walking',
                    '--end_time=5',
                    '--publish_rate=100',
                    '--w_swing_foot_x=%.1f' % x[0],
                    '--w_swing_foot_y=%.1f' % x[1],
                    '--w_swing_foot_z=%.1f' % x[2],
                    '--k_p_swing_foot_x=%.1f' % x[3],
                    '--k_p_swing_foot_y=%.1f' % x[4],
                    '--k_p_swing_foot_z=%.1f' % x[5],
                    '--k_d_swing_foot_x=%.1f' % x[6],
                    '--k_d_swing_foot_y=%.1f' % x[7],
                    '--k_d_swing_foot_z=%.1f' % x[8],
                    ]
  logger_process = subprocess.Popen(logger_cmd)
  simulation_process = subprocess.Popen(simulation_cmd)

  while simulation_process.poll() is None:  # while subprocess is alive
    time.sleep(1)
  logger_process.kill()

  # Get cost
  total_cost = 0

  log = lcm.EventLog(log_path, "r")
  x, u_meas, t_x, u, t_u, contact_info, contact_info_locs, t_contact_info, \
  osc_debug, fsm, estop_signal, switch_signal, t_controller_switch, t_pd, kp, kd, cassie_out, u_pd, t_u_pd, \
  osc_output, full_log = process_lcm_log.process_log(log, pos_map, vel_map,
    act_map, "CASSIE_INPUT")

  # tracking cost
  # TODO: need to fix the bug in error_y quaternioin (pelvis_balance_traj and pelvis_heading_traj)
  err = osc_debug['swing_ft_traj'].error_y
  total_cost += 5 * np.sum(np.multiply(err, err))
  err = osc_debug['lipm_traj'].error_y
  total_cost += np.sum(np.multiply(err[:, 2], err[:, 2]))
  err = osc_debug['pelvis_balance_traj'].error_y
  total_cost += np.sum(np.multiply(err[:, :3], err[:, :3]))
  err = osc_debug['swing_toe_traj'].error_y
  total_cost += np.sum(np.multiply(err, err))
  err = osc_debug['swing_hip_yaw_traj'].error_y
  total_cost += np.sum(np.multiply(err, err))
  err = osc_debug['pelvis_heading_traj'].error_y
  total_cost += np.sum(np.multiply(err[:, :3], err[:, :3]))

  # effort cost
  total_cost += np.sum(np.multiply(u / 1000, u / 1000))

  print(total_cost)

  # import pdb;
  # pdb.set_trace()

  return total_cost
  # return 1 + np.sum((x - np.pi) ** 2)


def main():
  # Read in gains from yaml
  with open('examples/Cassie/osc/osc_walking_gains.yaml') as f:
    my_dict = yaml.safe_load(f)

  # Initial guess
  x_init = 9 * [0]
  x_init[0] = my_dict['SwingFootW'][0]
  x_init[1] = my_dict['SwingFootW'][4]
  x_init[2] = my_dict['SwingFootW'][8]
  x_init[3] = my_dict['SwingFootKp'][0]
  x_init[4] = my_dict['SwingFootKp'][4]
  x_init[5] = my_dict['SwingFootKp'][8]
  x_init[6] = my_dict['SwingFootKd'][0]
  x_init[7] = my_dict['SwingFootKd'][4]
  x_init[8] = my_dict['SwingFootKd'][8]

  # Construct CMA
  sigma_init = 5
  es = cma.CMAEvolutionStrategy(x_init, sigma_init, {'popsize': 12})

  # Testing
  # obj_func(x_init)

  # Optimize
  start = time.time()
  es.optimize(obj_func, n_jobs=1)
  end = time.time()
  print("solve time = " + str(end - start))
  print()

  # es.result_pretty()

  # print(es.popsize)
  # print(es.opts)

  # import pdb; pdb.set_trace()


def reference_func():
  filename = FindResourceOrThrow(
    '../dairlib_data/goldilocks_models/find_models/robot_1/dircon_trajectory')

  dircon_traj = pydairlib.lcm_trajectory.DirconTrajectory(filename)

  # For saving figures
  global save_path
  import getpass
  username = getpass.getuser()
  save_path = "/home/" + username + "/"

  # Build MBP
  global plant, context, world, nq, nv, nx, nu
  builder = DiagramBuilder()
  plant, _ = AddMultibodyPlantSceneGraph(builder, 0.0)
  Parser(plant).AddModelFromFile(
    FindResourceOrThrow("examples/Cassie/urdf/cassie_v2.urdf"))
  plant.mutable_gravity_field().set_gravity_vector(-9.81 * np.array([0, 0, 1]))
  plant.Finalize()

  # Conext and world
  context = plant.CreateDefaultContext()
  world = plant.world_frame()
  global l_toe_frame, r_toe_frame
  global front_contact_disp, rear_contact_disp
  global l_loop_closure, r_loop_closure
  l_toe_frame = plant.GetBodyByName("toe_left").body_frame()
  r_toe_frame = plant.GetBodyByName("toe_right").body_frame()
  front_contact_disp = np.array((-0.0457, 0.112, 0))
  rear_contact_disp = np.array((0.088, 0, 0))
  l_loop_closure = LeftLoopClosureEvaluator(plant)
  r_loop_closure = RightLoopClosureEvaluator(plant)

  # MBP params
  nq = plant.num_positions()
  nv = plant.num_velocities()
  nx = plant.num_positions() + plant.num_velocities()
  nu = plant.num_actuators()


if __name__ == "__main__":
  main()
