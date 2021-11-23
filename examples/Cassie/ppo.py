import gym
import torch
import torch.nn as nn
from datetime import datetime
from stable_baselines3 import PPO

from cassie_joy_env import CassieEnv_Joystick

workspace = [[-1, 5], [-3, 3], [0.5, 1.3]]
env = CassieEnv_Joystick("CASSIE_VIRTUAL_RADIO", "CASSIE_STATE_SIMULATION", 10, workspace)

now = datetime.now()
logdir = "./rl_logging/" + now.strftime("%d%m%Y_%H%M%S_ppo") + "/"
model = PPO("MultiInputPolicy", env, verbose = 1, tensorboard_log = logdir, policy_kwargs=dict(normalize_images=False))
model.learn(10000)
model.save("ppo-cassie_joy")
