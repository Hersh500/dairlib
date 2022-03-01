from datetime import datetime
import yaml
import shutil
import argparse

import gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from cassie_joy_env import CassieEnv_Joystick

# Get the fname of the experiment config file
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="path to config file", required = True)
parser.add_argument("--model", type=str, help="path to pretrained model", required = False)
parser.add_argument("--eval", type=bool, help="evaluate the pretrained model (MUST specify a model)", required = False, default = False)
args = parser.parse_args()
config_fname = args.config
model_fname = args.model
to_eval = args.eval

# Read the config file
with open(config_fname, 'r') as f:
    params = yaml.safe_load(f)
env_params = params['env_params']
workspace = [env_params['x_lims'], env_params['y_lims'], env_params['z_lims']]
rate = env_params['rate_hz']
goal = env_params['goal_state']
viz = env_params['visualize']
terrain_class = env_params["terrain_class"]
num_features = env_params["num_features"]
reward_fn_type = env_params["reward_fn"]
use_acc_penalty = env_params["use_acc_penalty"]

env = CassieEnv_Joystick("CASSIE_VIRTUAL_RADIO", "CASSIE_STATE_RL", rate, workspace, goal, 
                         viz, terrain_class, num_features, reward_fn_type, use_acc_penalty)

now = datetime.now()
logdir = "./rl_logging/" + now.strftime("%d_%m_%Y_%H%M%S_ppo") + "/"

# copy the config file into the logging folder, so we know the parameters that were used
# for the experiment
if not to_eval:
    try:
        rl_params = params['ppo_params']
        if model_fname is not None:
            model = PPO.load(model_fname, env)
        else:
            model = PPO("MultiInputPolicy", env, verbose = rl_params["verbose"], tensorboard_log = logdir, policy_kwargs=dict(normalize_images=False), n_steps = rl_params["n_steps"], batch_size = rl_params["batch_size"])
        model.learn(rl_params['num_episodes'])
        model.save(logdir + "ppo-cassie_joy")
    except KeyboardInterrupt:
        pass
    shutil.copyfile(config_fname, logdir + "config.yaml", follow_symlinks=True)
else:
    print("Evaluating policy...")
    if model_fname is None:
        print("Error! If evaluating a policy, must specify a pretrained model to use")
    else:
        # use the same config that the model was trained in?
        model = PPO.load(model_fname, env)
        evaluate_policy(model, env, 10)
