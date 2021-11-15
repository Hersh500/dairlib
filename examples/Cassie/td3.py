# adapted from https://github.com/sfujim/TD3/blob/main/TD3.py
import copy
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import collections, random
import policies
from datetime import datetime
import argparse
from torch.utils.tensorboard import SummaryWriter
import pickle
from cassie_joy_env import CassieEnv_Joystick
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer():
    def __init__(self, device):
        self.buffer = collections.deque(maxlen=50000)
        self.device = device

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, t_lst, a_lst, r_lst, s_prime_lst, t_prime_lst, done_mask_lst = [], [], [], [], [], [],[]

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s[0])
            t_lst.append(s[1])
            a_lst.append(a)  # todo: not hardcode this

            r_lst.append([r])
            s_prime_lst.append(s_prime[0])
            t_prime_lst.append(s_prime[1])
            done_mask = 0.0 if done else 1.0 
            done_mask_lst.append([done_mask])
        
        return (torch.tensor(s_lst, dtype=torch.float).to(self.device), torch.tensor(t_lst, dtype=torch.float).to(self.device)), torch.tensor(a_lst, dtype=torch.float).to(self.device), \
                torch.tensor(r_lst, dtype=torch.float).to(self.device), (torch.tensor(s_prime_lst, dtype=torch.float).to(self.device), torch.tensor(t_prime_lst, dtype=torch.float).to(self.device)), \
                torch.tensor(done_mask_lst, dtype=torch.float).to(self.device)
    
    def size(self):
        return len(self.buffer)

class TD3(object):
    def __init__(self,
        state_dim,
        image_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2):

        state_enc_pi = policies.StateEncoder(policies.VisionFrontend2D(image_dim),
                                             state_dim,
                                             state_encoding_dim = 32,
                                             output_dim = 64) 
        self.actor = policies.Actor_TD3(state_enc_pi, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = 1e-4)

        state_enc_q1 = policies.StateEncoder(policies.VisionFrontend2D(image_dim),
                                             state_dim,
                                             state_encoding_dim = 32,
                                             output_dim = 64) 
        state_enc_q2 = copy.deepcopy(state_enc_q1)
        self.critic = policies.Critic_TD3(state_enc_q1, state_enc_q2, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = 1e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0


    def select_action(self, state):
        return self.actor(state).cpu().data.numpy().flatten()


    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer 
        state, action, reward, next_state, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


# runs the training loop
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--info", type=str)
    args = parser.parse_args()
    meta_info = args.info

    now = datetime.now()
    logdir = now.strftime("%d%m%Y_%H%M%S_td3") + "/"
    writer = SummaryWriter(log_dir = "./rl_logging/" + logdir)

    # define the hyperparameters
    start_timesteps = 512
    eval_freq = 2e3
    max_timesteps = 10000
    expl_noise = 0.3
    batch_size = 128
    discount = 0.99 
    tau = 0.005
    policy_noise = 0.3
    noise_clip = 0.5
    policy_freq = 2

    # Set seeds
    # env.seed(args.seed)
    # env.action_space.seed(args.seed)
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    workspace = [[-1, 5], [-3, 3], [0.5, 1.3]]
    env = CassieEnv_Joystick("CASSIE_VIRTUAL_RADIO", "CASSIE_STATE_SIMULATION", 10, workspace)
    
    state_dim = env.state_dim
    image_dim = env.image_dim
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    # Initialize policy
    policy = TD3(state_dim, 
                 image_dim,
                 action_dim,
                 max_action,
                 discount,
                 tau,
                 policy_noise,
                 noise_clip,
                 policy_freq)

    replay_buffer = ReplayBuffer(device)
    

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    try:
        for t in range(int(max_timesteps)):
            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < start_timesteps:
                action = env.action_space.sample()
            else:
                state_tensor = torch.from_numpy(state[0]).float().to(device)
                image_tensor = torch.from_numpy(state[1]).float().to(device)
                image_tensor = image_tensor.view(1, 1, image_tensor.size(0), image_tensor.size(1))
                action = (
                    policy.select_action((state_tensor, image_tensor))
                    + np.random.normal(0, max_action * expl_noise, size=action_dim)
                ).clip(-max_action, max_action)

            # Perform action
            next_state, reward, done = env.step(action)
            done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

            # Store data in replay buffer
            replay_buffer.put((state, action, reward, next_state, done_bool))

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            # TODO: is this fast enough for realtime?
            if t >= start_timesteps:
                t0 = time.time()
                policy.train(replay_buffer, batch_size)
                t1 = time.time()
                

            if done: 
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                # Reset environment
                state, done = env.reset(), False
                print("RESET STATE:", state[0])
                writer.add_scalar("Loss/training_reward", episode_reward, episode_num)
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
                print("-----------On episode:", episode_num, "-----------")

            if (t + 1) % eval_freq == 0:
                policy.save("./logging/" + logdir + "policy")
                if meta_info is not None:
                    f = open("./logging/" + logdir + "info.txt", 'w')
                    f.write(meta_info)
                    f.close()

    except Exception as e:
        print(e)
        print("exiting because of error")
        env.kill_procs()
        env.kill_director()

if __name__ == "__main__":
    main()
