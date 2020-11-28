import cma  # https://github.com/CMA-ES/pycma
import subprocess
import time
import math
import numpy as np
import random
import lcm
import sys
import yaml
from random import randrange
import pdb
import os
from shutil import copyfile
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

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


# TODO: check each term of the cost and potentially save them
# TODO: avoid negative gains (can we add constraints? otherwise add it to cost)
# TODO: run one lcm-logger per channel name

# TODO: make the termination condition looser (sigma can be 5e-3)

# TODO fix the bug in swing foot desired traj when fsm switching

# TODO: Can probably add noise and add delay to the simulation

# TODO: check if lcm-logger could miss data when data rate is high and when doing multithreading

# TODO: can try effort dot to cost

def obj_func(x):
  sample_id = ""
  if not save_log:
    for i in range(10):
      sample_id += str(randrange(10))

  # Scale gains
  gains = param_dim * [0]
  gains[0] = x[0] * yaml_gains['w_accel']
  gains[1] = x[1] * yaml_gains['w_soft_constraint']
  gains[2] = x[2] * yaml_gains['w_swing_toe']
  gains[3] = x[3] * yaml_gains['swing_toe_kp']
  gains[4] = x[4] * yaml_gains['swing_toe_kd']
  gains[5] = x[5] * yaml_gains['w_hip_yaw']
  gains[6] = x[6] * yaml_gains['hip_yaw_kp']
  gains[7] = x[7] * yaml_gains['hip_yaw_kd']
  gains[8] = x[8] * yaml_gains['CoMW'][8]
  gains[9] = x[9] * yaml_gains['CoMKp'][8]
  gains[10] = x[10] * yaml_gains['CoMKd'][8]
  gains[11] = x[11] * yaml_gains['PelvisBalanceW'][0]
  gains[12] = x[12] * yaml_gains['PelvisBalanceW'][4]
  gains[13] = x[13] * yaml_gains['PelvisBalanceKp'][0]
  gains[14] = x[14] * yaml_gains['PelvisBalanceKp'][4]
  gains[15] = x[15] * yaml_gains['PelvisBalanceKd'][0]
  gains[16] = x[16] * yaml_gains['PelvisBalanceKd'][4]
  gains[17] = x[17] * yaml_gains['PelvisHeadingW'][8]
  gains[18] = x[18] * yaml_gains['PelvisHeadingKp'][8]
  gains[19] = x[19] * yaml_gains['PelvisHeadingKd'][8]
  gains[20] = x[20] * yaml_gains['SwingFootW'][0]
  gains[21] = x[21] * yaml_gains['SwingFootW'][4]
  gains[22] = x[22] * yaml_gains['SwingFootW'][8]
  gains[23] = x[23] * yaml_gains['SwingFootKp'][0]
  gains[24] = x[24] * yaml_gains['SwingFootKp'][4]
  gains[25] = x[25] * yaml_gains['SwingFootKp'][8]
  gains[26] = x[26] * yaml_gains['SwingFootKd'][0]
  gains[27] = x[27] * yaml_gains['SwingFootKd'][4]
  gains[28] = x[28] * yaml_gains['SwingFootKd'][8]

  # initialize cost
  cost = 0

  # Penalize negative weights
  for i in range(param_dim):
    cost += 100 * (math.exp(max(0, -x[i])) - 1)
    if gains[i] < 0:
      cost += 499

  if cost == 0:
    n_trail_for_random_spring = (1 if save_log else 9)
    for i in range(n_trail_for_random_spring):
      cost = run_sim_and_eval_cost(cost, gains, sample_id)
    cost /= n_trail_for_random_spring

  print("cost = " + str(cost))
  return cost


def run_sim_and_eval_cost(cost, gains, sample_id):
  # Randomize spring stiffness
  spring_stiffness = (default_spring_stiffness if save_log
                      else [random.uniform(800, 2200) for i in range(4)])

  # Randomize initial pelvis disturbance
  pelvis_disturbance = ([0, 0, 0] if save_log
                        else [random.uniform(-1, 1) for i in range(3)])

  # Run the simulation and lcm-logger
  log_path = dir + 'testlog' + str(sample_id)
  logger_cmd = ['lcm-logger',
                '-f',
                '--quiet',
                '%s' % log_path,
                ]
  simulation_cmd = \
    ['bazel-bin/examples/Cassie/run_sim_and_walking',
     '--sample_id=%s' % sample_id,
     '--end_time=5',
     '--publish_rate=100',
     '--target_realtime_rate=10',
     '--print_gains=' + str(save_log).lower(),
     '--knee_spring_left=%.2f' % spring_stiffness[0],
     '--knee_spring_right=%.2f' % spring_stiffness[1],
     '--ankle_spring_left=%.2f' % spring_stiffness[2],
     '--ankle_spring_right=%.2f' % spring_stiffness[3],
     '--pelvis_disturbnace_xdot=%.2f' % pelvis_disturbance[0],
     '--pelvis_disturbnace_ydot=%.2f' % pelvis_disturbance[1],
     '--pelvis_disturbnace_zdot=%.2f' % pelvis_disturbance[2],
     '--w_accel=%.8f' % gains[0],
     '--w_soft_constraint=%.2f' % gains[1],
     '--w_swing_toe=%.2f' % gains[2],
     '--swing_toe_kp=%.2f' % gains[3],
     '--swing_toe_kd=%.2f' % gains[4],
     '--w_hip_yaw=%.2f' % gains[5],
     '--hip_yaw_kp=%.2f' % gains[6],
     '--hip_yaw_kd=%.2f' % gains[7],
     '--w_com_z=%.2f' % gains[8],
     '--k_p_com_z=%.2f' % gains[9],
     '--k_d_com_z=%.2f' % gains[10],
     '--w_pelvis_balance_x=%.2f' % gains[11],
     '--w_pelvis_balance_y=%.2f' % gains[12],
     '--k_p_pelvis_balance_x=%.2f' % gains[13],
     '--k_p_pelvis_balance_y=%.2f' % gains[14],
     '--k_d_pelvis_balance_x=%.2f' % gains[15],
     '--k_d_pelvis_balance_y=%.2f' % gains[16],
     '--w_pelvis_heading_z=%.2f' % gains[17],
     '--k_p_pelvis_heading_z=%.2f' % gains[18],
     '--k_d_pelvis_heading_z=%.2f' % gains[19],
     '--w_swing_foot_x=%.2f' % gains[20],
     '--w_swing_foot_y=%.2f' % gains[21],
     '--w_swing_foot_z=%.2f' % gains[22],
     '--k_p_swing_foot_x=%.2f' % gains[23],
     '--k_p_swing_foot_y=%.2f' % gains[24],
     '--k_p_swing_foot_z=%.2f' % gains[25],
     '--k_d_swing_foot_x=%.2f' % gains[26],
     '--k_d_swing_foot_y=%.2f' % gains[27],
     '--k_d_swing_foot_z=%.2f' % gains[28],
     (' | tee -a ' + dir + 'gains' if save_log else ''),
     ]
  if save_log:
    simulation_cmd = ' '.join(simulation_cmd)
  logger_process = subprocess.Popen(logger_cmd)
  simulation_process = subprocess.Popen(simulation_cmd, shell=save_log)

  while simulation_process.poll() is None:  # while subprocess is alive
    time.sleep(1)
  logger_process.kill()

  # Compute cost
  log = lcm.EventLog(log_path, "r")
  x, u_meas, t_x, u, t_u, contact_info, contact_info_locs, t_contact_info, \
  osc_debug, fsm, estop_signal, switch_signal, t_controller_switch, t_pd, kp, kd, cassie_out, u_pd, t_u_pd, \
  osc_output, full_log = process_lcm_log.process_log(log, pos_map, vel_map,
    act_map, "___", sample_id)

  try:
    # TODO: need to fix the bug in error_y quaternioin (pelvis_balance_traj and pelvis_heading_traj)
    # tracking cost
    w_vel = 0  # 1/400
    err = osc_debug['swing_ft_traj'].error_y
    swing_ft_error_y_cost = 5 * np.sum(np.multiply(err, err))
    cost += swing_ft_error_y_cost
    err = osc_debug['swing_ft_traj'].error_ydot
    swing_ft_error_ydot_cost = 5 * w_vel * np.sum(np.multiply(err, err))
    cost += swing_ft_error_ydot_cost
    err = osc_debug['lipm_traj'].error_y
    lipm_error_y_cost = np.sum(np.multiply(err[:, 2], err[:, 2]))
    cost += lipm_error_y_cost
    err = osc_debug['pelvis_balance_traj'].error_y
    pelvis_balance_error_y_cost = np.sum(np.multiply(err[:, :3], err[:, :3]))
    cost += pelvis_balance_error_y_cost
    err = osc_debug['swing_toe_traj'].error_y
    swing_toe_error_y_cost = np.sum(np.multiply(err, err))
    cost += swing_toe_error_y_cost
    err = osc_debug['swing_toe_traj'].error_ydot
    swing_toe_error_ydot_cost = w_vel * np.sum(np.multiply(err, err))
    cost += swing_toe_error_ydot_cost
    err = osc_debug['swing_hip_yaw_traj'].error_y
    swing_hip_yaw_error_y_cost = np.sum(np.multiply(err, err))
    cost += swing_hip_yaw_error_y_cost
    err = osc_debug['swing_hip_yaw_traj'].error_ydot
    swing_hip_yaw_error_ydot_cost = w_vel * np.sum(np.multiply(err, err))
    cost += swing_hip_yaw_error_ydot_cost
    err = osc_debug['pelvis_heading_traj'].error_ydot
    pelvis_heading_error_ydot_cost = w_vel * np.sum(
      np.multiply(err[:, 2], err[:, 2]))
    cost += pelvis_heading_error_ydot_cost

    # effort cost
    effort_cost = np.sum(np.multiply(u / 5000, u / 5000))
    cost += effort_cost

    # spring acceleration cost
    spring_cost = 0.0
    t_diff = np.diff(t_x)
    for name in spring_joint_names:
      vel_diff = np.diff(x[:, nq + vel_map[name]])
      vel_dot = vel_diff / t_diff
      spring_cost += np.sum(np.multiply(vel_dot / 4000, vel_dot / 4000))
    cost += spring_cost

    # d gain < 10 cost
    d_gain_cost = 0.0
    # doesn't include toe joint
    for idx in [7, 10, 15, 16, 19, 26, 27, 28]:
      d_gain_cost += math.exp(max(0, gains[idx] - 10)) - 1
    cost += d_gain_cost

    if save_log:
      print("Cost breakdown: ")
      print("swing_ft_error_y: " + str(swing_ft_error_y_cost))
      print("swing_ft_error_ydot: " + str(swing_ft_error_ydot_cost))
      print("lipm_error_y: " + str(lipm_error_y_cost))
      print("pelvis_balance_error_y: " + str(pelvis_balance_error_y_cost))
      print("swing_toe_error_y: " + str(swing_toe_error_y_cost))
      print("swing_toe_error_ydot: " + str(swing_toe_error_ydot_cost))
      print("swing_hip_yaw_error_y: " + str(swing_hip_yaw_error_y_cost))
      print("swing_hip_yaw_error_ydot: " + str(swing_hip_yaw_error_ydot_cost))
      print("pelvis_heading_error_ydot: " + str(pelvis_heading_error_ydot_cost))
      print("effort: " + str(effort_cost))
      print("spring_cost: " + str(spring_cost))
      print("d_gain_cost: " + str(d_gain_cost))
  except:
    # There could be missing trajs when simulation terminates early
    cost += 1000

  # Delete log file
  if save_log:
    copyfile(log_path, dir + time.ctime())
  os.remove(log_path)

  return cost


def main():
  # Settings
  global domain_randomization, dir
  domain_randomization = False
  dir = "../dairlib_data/cassie_cma/"
  n_theads = 12

  # Parameters
  global param_dim, popsize
  param_dim = 29
  popsize = param_dim * 2

  # Create folder
  Path(dir).mkdir(parents=True, exist_ok=True)

  # Build MBP
  global plant, pos_map, vel_map, act_map, default_spring_stiffness
  default_spring_stiffness = [1500, 1500, 1250, 1250]
  pydrake.common.set_log_level("err")
  builder = DiagramBuilder()
  plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
  pydairlib.cassie.cassie_utils.addCassieMultibody(plant, scene_graph, True,
    "examples/Cassie/urdf/cassie_v2.urdf", False, False,
    default_spring_stiffness)
  plant.Finalize()

  pos_map = pydairlib.multibody.makeNameToPositionsMap(plant)
  vel_map = pydairlib.multibody.makeNameToVelocitiesMap(plant)
  act_map = pydairlib.multibody.makeNameToActuatorsMap(plant)

  # Some setups
  global spring_joint_names, nq
  spring_joint_names = ["knee_joint_leftdot",
                        "knee_joint_rightdot",
                        "ankle_spring_joint_leftdot",
                        "ankle_spring_joint_rightdot"]
  nq = plant.num_positions()

  # Read in gains from yaml
  global yaml_gains
  with open('examples/Cassie/osc/osc_walking_gains.yaml') as f:
    yaml_gains = yaml.safe_load(f)
  copyfile("examples/Cassie/osc/osc_walking_gains.yaml",
    dir + "osc_walking_gains.yaml")

  # Initial guess
  x_init = param_dim * [1.0]
  # x_init[0] = 0.273
  # x_init[1] = 0.827
  # x_init[2] = 0.429
  # x_init[3] = 0.664
  # x_init[4] = 2.520
  # x_init[5] = 2.241
  # x_init[6] = 1.330
  # x_init[7] = 0.794
  # x_init[8] = 2.526
  # x_init[9] = 0.207
  # x_init[10] = 0.983
  # x_init[11] = 0.965
  # x_init[12] = 1.266
  # x_init[13] = 1.209
  # x_init[14] = 1.292
  # x_init[15] = 1.925
  # x_init[16] = 0.402
  # x_init[17] = 1.590
  # x_init[18] = 1.939
  # x_init[19] = 3.497
  # x_init[20] = 2.021
  # x_init[21] = 1.598
  # x_init[22] = 0.299
  # x_init[23] = 2.195
  # x_init[24] = 1.023
  # x_init[25] = 1.762
  # x_init[26] = 0.979
  # x_init[27] = 0.935
  # x_init[28] = 0.998

  # Construct CMA
  sigma_init = 0.5
  es = cma.CMAEvolutionStrategy(x_init, sigma_init, {'popsize': popsize})

  # Save the initial log
  global save_log
  save_log = True
  obj_func(x_init)
  save_log = False

  # Optimize
  es.optimize(obj_func, n_jobs=n_theads)
  es.result_pretty()

  # Save the log of the best solution
  save_log = True
  obj_func(es.result.xbest.tolist())
  save_log = False

  print(es.popsize)
  print(es.opts)
  pdb.set_trace()


if __name__ == "__main__":
  main()
