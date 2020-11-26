import cma  # https://github.com/CMA-ES/pycma
import subprocess
import time
import numpy as np
import lcm
import sys
import yaml
import pdb
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


# (seems to be fine now) TODO: need to handle error (stop lcm-logger)
# TODO: avoid negative gains (can we add constraints? otherwise add it to cost)
# TODO: now you can use global variable, you can multithread this (you can specify the channels to listen to).
# (don't need this, cause there is only one cma outer-loop) TODO: Also, need to check how to change output files' names
# TODO: run one lcm-logger per channel name

# TODO: make the termination condition looser (sigma can be 5e-3)

# TODO fix the bug in swing foot desired traj when fsm switching

# TODO: Can probably add noise and add delay to the simulation

def obj_func(x):
  global sample_idx
  # sample_idx_in_this_iteration = sample_idx % popsize
  sample_idx_in_this_iteration = 0
  sample_idx = sample_idx + 1

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

  log_path = dir + 'testlog' + str(sample_idx_in_this_iteration)
  logger_cmd = ['lcm-logger',
                '-f',
                '--quiet',
                '%s' % log_path,
                ]
  simulation_cmd = ['bazel-bin/examples/Cassie/run_sim_and_walking',
                    '--sample_idx=%d' % sample_idx_in_this_iteration,
                    '--end_time=5',
                    '--publish_rate=100',
                    '--target_realtime_rate=10',
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
    act_map, "___", sample_idx_in_this_iteration)

  # tracking cost
  # TODO: need to fix the bug in error_y quaternioin (pelvis_balance_traj and pelvis_heading_traj)
  try:
    err = osc_debug['swing_ft_traj'].error_y
    total_cost += 5 * np.sum(np.multiply(err, err))
    err = osc_debug['swing_ft_traj'].error_ydot
    total_cost += 5 * 1 / 400 * np.sum(np.multiply(err, err))
    err = osc_debug['lipm_traj'].error_y
    total_cost += np.sum(np.multiply(err[:, 2], err[:, 2]))
    err = osc_debug['pelvis_balance_traj'].error_y
    total_cost += np.sum(np.multiply(err[:, :3], err[:, :3]))
    err = osc_debug['swing_toe_traj'].error_y
    total_cost += np.sum(np.multiply(err, err))
    err = osc_debug['swing_toe_traj'].error_ydot
    total_cost += 1 / 400 * np.sum(np.multiply(err, err))
    err = osc_debug['swing_hip_yaw_traj'].error_y
    total_cost += np.sum(np.multiply(err, err))
    err = osc_debug['swing_hip_yaw_traj'].error_ydot
    total_cost += 1 / 400 * np.sum(np.multiply(err, err))
    err = osc_debug['pelvis_heading_traj'].error_y
    total_cost += np.sum(np.multiply(err[:, :3], err[:, :3]))

    # effort cost
    total_cost += np.sum(np.multiply(u / 1000, u / 1000))
  except:
    # There could be missing trajs when simulation terminates early
    # TODO: check if lcm-logger could miss data when data rate is high
    total_cost = 1000

  # print("sample #" + str(sample_idx) + ", total_cost = " + str(total_cost))
  return total_cost


def main():
  # Settings
  global domain_randomization, dir
  domain_randomization = False
  dir = "../dairlib_data/cassie_cma/"

  # Parameters
  global param_dim, popsize
  param_dim = 29
  popsize = param_dim * 2
  # popsize = 3

  # Create folder
  Path(dir).mkdir(parents=True, exist_ok=True)

  # Build MBP
  global plant, pos_map, vel_map, act_map
  pydrake.common.set_log_level("err")
  builder = DiagramBuilder()
  plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
  pydairlib.cassie.cassie_utils.addCassieMultibody(plant, scene_graph, True,
    "examples/Cassie/urdf/cassie_v2.urdf", False, False)
  plant.Finalize()

  pos_map = pydairlib.multibody.makeNameToPositionsMap(plant)
  vel_map = pydairlib.multibody.makeNameToVelocitiesMap(plant)
  act_map = pydairlib.multibody.makeNameToActuatorsMap(plant)

  # Read in gains from yaml
  global yaml_gains
  with open('examples/Cassie/osc/osc_walking_gains.yaml') as f:
    yaml_gains = yaml.safe_load(f)
  copyfile("examples/Cassie/osc/osc_walking_gains.yaml",
    dir + "osc_walking_gains.yaml")

  # Initial guess
  x_init = param_dim * [1]

  # Construct CMA
  sigma_init = 0.2
  es = cma.CMAEvolutionStrategy(x_init, sigma_init, {'popsize': popsize})
  global sample_idx
  sample_idx = 0

  # Testing
  # obj_func(x_init)

  # Optimize
  es.optimize(obj_func, n_jobs=1)
  es.result_pretty()

  print(es.popsize)
  print(es.opts)
  pdb.set_trace()


if __name__ == "__main__":
  main()
