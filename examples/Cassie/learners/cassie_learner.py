## Generic Script for training RL policies for various Cassie Tasks

from datetime import datetime
import yaml
import shutil
import argparse
import gym
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

import cassie_joy_env


def getEnv(env_name, env_params):
    radio_channel = "CASSIE_VIRTUAL_RADIO"
    state_channel = "CASSIE_STATE_RL" 
    if env_name == "fixed_goal":
        return cassie_joy_env.Cassie_FixedGoal_Blind(radio_channel, state_channel, env_params["rate_hz"],
                                      [env_params['x_lims'], env_params['y_lims'], env_params['z_lims']],
                                      env_params['goal_state'],
                                      env_params['visualize']), "MlpPolicy"

    elif env_name == "rand_goal":
        return cassie_joy_env.Cassie_RandGoal_Blind(radio_channel, state_channel, env_params["rate_hz"],
                                      [env_params['x_lims'], env_params['y_lims'], env_params['z_lims']],
                                      env_params['visualize']), "MlpPolicy"

    elif env_name == "rand_goal_obst":
        return cassie_joy_env.Cassie_RandGoalObst_Blind(radio_channel, state_channel, env_params["rate_hz"],
                                      [env_params['x_lims'], env_params['y_lims'], env_params['z_lims']],
                                      env_params['visualize']), "MlpPolicy"
    elif env_name == "fixed_goal_depth":
        return cassie_joy_env.Cassie_FixedGoal_Depth(radio_channel, state_channel, env_params["rate_hz"], 
                                                    [env_params['x_lims'], env_params['y_lims'], env_params['z_lims']],
                                                    env_params["goal_state"],
                                                    env_params['visualize']), "MultiInputPolicy"
    else:
        raise NotImplementedError("This environment has not been added.")

        
def loadModel(learner_name, fname, env):
    print("Loading pretrained model")
    if learner_name == "sac":
        model = SAC.load(fname, env)
    elif learner_name == "ppo":
        model = PPO.load(fname, env)
    return model


def getModel(learner_name, policy_type, env, rl_params, logdir):
    if learner_name == "sac":
        model = SAC(policy_type,
                    env, verbose = rl_params["verbose"],
                    tensorboard_log = logdir,
                    # policy_kwargs=dict(normalize_images=False),
                    learning_starts = rl_params["learning_starts"],
                    batch_size = rl_params["batch_size"],
                    buffer_size = rl_params["buffer_size"])
    elif learner_name == "ppo":
        model = PPO(policy_type,
                    env,
                    verbose = rl_params["verbose"],
                    tensorboard_log = logdir,
                    # policy_kwargs=dict(normalize_images=False),
                    n_steps = rl_params["ppo_n_steps_per_update"],
                    batch_size = rl_params["batch_size"])
    return model


def train(learner, env, rl_params):
    learner.learn(rl_params['num_steps'])


def evaluate(model, env, num_episodes):
    evaluate_policy(model, env, num_episodes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to config file", required = True)
    parser.add_argument("--task", type=str, help="Task Name", required = True)
    parser.add_argument("--learner_type", type=str, help="Learner type {sac, ppo}", required = True)
    parser.add_argument("--model", type=str, help="path to pretrained model", required = False)
    parser.add_argument("--eval", type=bool, help="evaluate the pretrained model (MUST specify a model)", required = False, default = False)

    args = parser.parse_args()
    task_name = args.task
    learner_name = args.learner_type
    config_fname = args.config
    model_fname = args.model
    to_eval = args.eval

    with open(config_fname, 'r') as f:
        params = yaml.safe_load(f)

    rl_params = params['rl_params']
    env_params = params['env_params']
    env, policy_type = getEnv(task_name, env_params)
    desc = learner_name + "_" + task_name
    logdir = "./rl_logging/" + datetime.now().strftime("%b_%d_%Y_%H%M") + desc +  "/"

    if to_eval:
        model = loadModel(learner_name, model_fname, env)
        evaluate(model, env, 10) 
    else:
        if model_fname is None:
            model = getModel(learner_name, policy_type, env, rl_params, logdir)
        else:
            model = loadModel(learner_name, model_fname, env)
        try:
            train(model, env, rl_params)
            model.save(logdir + desc)
            shutil.copyfile(config_fname, logdir + "config.yaml", follow_symlinks=True)
        except KeyboardInterrupt:
            model.save(logdir + desc)
            shutil.copyfile(config_fname, logdir + "config.yaml", follow_symlinks=True)

if __name__ == "__main__":
    main()
