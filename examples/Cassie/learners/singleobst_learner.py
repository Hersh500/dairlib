from datetime import datetime
import yaml
import shutil
import argparse
import gym
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

from cassie_joy_env import Cassie_SingleObst, Cassie_FixedGoal_Blind, Cassie_RandGoal_Blind

# Get the fname of the experiment config file
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="path to config file", required = True)
parser.add_argument("--model", type=str, help="path to pretrained model", required = False)
parser.add_argument("--eval", type=bool, help="evaluate the pretrained model (MUST specify a model)", required = False, default = False)
args = parser.parse_args()
config_fname = args.config
model_fname = args.model
to_eval = args.eval

with open(config_fname, 'r') as f:
    params = yaml.safe_load(f)
env_params = params['env_params']
workspace = [env_params['x_lims'], env_params['y_lims'], env_params['z_lims']]
rate = env_params['rate_hz']
goal = env_params['goal_state']
viz = env_params['visualize']
blind = env_params['blind']

if not blind:
    env = Cassie_SingleObst("CASSIE_VIRTUAL_RADIO", "CASSIE_STATE_RL", rate, workspace, goal, 
                             viz)
    policy_type = "MultiInputPolicy"
else:
    print("Warning: not using image inputs")
    env = Cassie_FixedGoal_Blind("CASSIE_VIRTUAL_RADIO", "CASSIE_STATE_RL", rate, workspace, goal,
                                  viz) 
    policy_type = "MlpPolicy"
    

now = datetime.now()
logdir = "./rl_logging/" + now.strftime("%d_%m_%Y_%H%M%S_sac") + "/"

if not to_eval:
    try:
        rl_params = params['sac_params']
        if model_fname is not None:
            model = SAC.load(model_fname, env)
        else:
            model = SAC(policy_type, env, verbose = rl_params["verbose"], tensorboard_log = logdir, policy_kwargs=dict(normalize_images=False), learning_starts = rl_params["learning_starts"], batch_size = rl_params["batch_size"], buffer_size = rl_params["buffer_size"])
        model.learn(total_timesteps = rl_params['num_steps'])
        model.save(logdir + "sac-cassie_singleobst")
    except KeyboardInterrupt:
        model.save(logdir + "sac-cassie_singleobst")
        pass
    shutil.copyfile(config_fname, logdir + "config.yaml", follow_symlinks=True)
else:
    print("Evaluating policy...")
    if model_fname is None:
        print("Error! If evaluating a policy, must specify a pretrained model to use")
    else:
        # use the same config that the model was trained in?
        model = SAC.load(model_fname, env)
        evaluate_policy(model, env, 10)
