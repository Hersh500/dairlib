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


# (done) TODO: learn how to read the output files and how to see the solution history
# TODO: avoid negative gains (can we add constraints? otherwise add it to cost)
# TODO: need to handle error (stop lcm-logger)
# TODO: need suppress lcmlogger message
# TODO: now you can use global variable, you can multithread this (you can specify the channels to listen to)

# TODO: Can probably add noise and add delay to the simulation

def obj_func(x):
  gains = 9 * [0]
  gains[0] = x[0] * yaml_gains['SwingFootW_scale']
  gains[1] = x[1] * yaml_gains['SwingFootW_scale']
  gains[2] = x[2] * yaml_gains['SwingFootW_scale']
  gains[3] = x[3] * yaml_gains['SwingFootKp_scale']
  gains[4] = x[4] * yaml_gains['SwingFootKp_scale']
  gains[5] = x[5] * yaml_gains['SwingFootKp_scale']
  gains[6] = x[6] * yaml_gains['SwingFootKd_scale']
  gains[7] = x[7] * yaml_gains['SwingFootKd_scale']
  gains[8] = x[8] * yaml_gains['SwingFootKd_scale']

  log_path = 'testlog'
  logger_cmd = ['lcm-logger',
                '-f',
                '%s' % log_path,
                ]
  simulation_cmd = ['bazel-bin/examples/Cassie/run_sim_and_walking',
                    '--end_time=5',
                    '--publish_rate=100',
                    '--w_swing_foot_x=%.1f' % gains[0],
                    '--w_swing_foot_y=%.1f' % gains[1],
                    '--w_swing_foot_z=%.1f' % gains[2],
                    '--k_p_swing_foot_x=%.1f' % gains[3],
                    '--k_p_swing_foot_y=%.1f' % gains[4],
                    '--k_p_swing_foot_z=%.1f' % gains[5],
                    '--k_d_swing_foot_x=%.1f' % gains[6],
                    '--k_d_swing_foot_y=%.1f' % gains[7],
                    '--k_d_swing_foot_z=%.1f' % gains[8],
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

  global sample_idx
  sample_idx = sample_idx + 1
  print("sample #" + str(sample_idx) + ", total_cost = " + str(total_cost))

  return total_cost


def main():
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

  # Initial guess
  x_init = 9 * [0]
  x_init[0] = yaml_gains['SwingFootW'][0] / yaml_gains['SwingFootW_scale']
  x_init[1] = yaml_gains['SwingFootW'][4] / yaml_gains['SwingFootW_scale']
  x_init[2] = yaml_gains['SwingFootW'][8] / yaml_gains['SwingFootW_scale']
  x_init[3] = yaml_gains['SwingFootKp'][0] / yaml_gains['SwingFootKp_scale']
  x_init[4] = yaml_gains['SwingFootKp'][4] / yaml_gains['SwingFootKp_scale']
  x_init[5] = yaml_gains['SwingFootKp'][8] / yaml_gains['SwingFootKp_scale']
  x_init[6] = yaml_gains['SwingFootKd'][0] / yaml_gains['SwingFootKd_scale']
  x_init[7] = yaml_gains['SwingFootKd'][4] / yaml_gains['SwingFootKd_scale']
  x_init[8] = yaml_gains['SwingFootKd'][8] / yaml_gains['SwingFootKd_scale']

  # Construct CMA
  sigma_init = 1
  es = cma.CMAEvolutionStrategy(x_init, sigma_init, {'popsize': 12})

  # Testing
  # obj_func(x_init)

  # Optimize
  global sample_idx
  sample_idx = -1
  start = time.time()
  es.optimize(obj_func, n_jobs=1)
  end = time.time()
  print("solve time = " + str(end - start))
  print()

  es.result_pretty()

  print(es.popsize)
  print(es.opts)

  import pdb;
  pdb.set_trace()


if __name__ == "__main__":
  main()
