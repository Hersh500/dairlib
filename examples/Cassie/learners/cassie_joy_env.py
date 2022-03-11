import lcm
from dairlib import lcmt_robot_output, lcmt_radio_out, lcmt_image_array, lcmt_image, lcmt_rl_step
import subprocess as sp
import gym
from gym import spaces 
import queue
import numpy as np
import time
import csv
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import threading
import argparse
import warnings

# this is bad, but need to suppress like 50 LCM deprecation warnings
warnings.filterwarnings("ignore")

class CassieEnv_Joystick(gym.Env):
    def __init__(self,
                action_channel,
                state_channel,
                rate,
                workspace,
                goal_state,
                visualize,
                terrain_class,
                num_features,
                reward_fn_type,
                acc_penalty):

        super(CassieEnv_Joystick, self).__init__()

        ### DRAKE SIMULATION STUFF ###
        # spawn the controller, and keep track of the pid
        self.ctrlr = None
        # spawn the simulation, and keep track of the pid (to kill to reset the sim)
        self.sim = None
        self.bin_dir = "./bazel-bin/examples/Cassie/"
        self.controller_p = "run_osc_walking_controller"
        self.simulation_p = "rl_multibody_sim"
        self.viz = visualize
        self.num_features = num_features
        self.terrain_class = terrain_class

        ### Communication Stuff ###
        self.lc = lcm.LCM()
        self.stop_listener = threading.Event()
        self.sub_state = self.lc.subscribe(state_channel, self.state_handler)
        self.sub_images = self.lc.subscribe("DRAKE_RGBD_CAMERA_IMAGES", self.image_handler)
        self.state_queue = queue.LifoQueue()
        self.image_queue = queue.LifoQueue()
        self.action_channel = action_channel
        self.state_channel = state_channel
        self.pos_names = ['base_qw', 'base_qx', 'base_qy', 'base_qz', 'base_x', 'base_y', 'base_z', 'hip_roll_left', 'hip_roll_right', 'hip_yaw_left', 'hip_yaw_right', 'hip_pitch_left', 'hip_pitch_right', 'knee_left', 'knee_right', 'knee_joint_left', 'knee_joint_right', 'ankle_joint_left', 'ankle_joint_right', 'ankle_spring_joint_left', 'toe_left', 'ankle_spring_joint_right', 'toe_right']
        self.ctrlr_options = ["--use_radio=1", "--cassie_out_channel=CASSIE_OUTPUT", "--channel_x=CASSIE_STATE_SIMULATION"]
        self.sim_ticks = 0

        ### Setting RL variables ###
        self.workspace = workspace
        self.goal_dist = np.linalg.norm(goal_state) 
        self.goal_state = goal_state
        self.done = False
        self.rate = rate
        self.image_dim = (64, 64, 1)
        ## will never want to go backward
        self.action_space = spaces.Box(low = np.array([0, 0, -1, -1]), high = np.array([1, 1, 1, 1]))
        # Added time as a state variable.
        self.observation_space = spaces.Dict({"position": spaces.Box(low = np.array([0, 0, 0, 0, 0, 0]), high = np.array([1, 1, 1, 1, 1, 1])),
                                              "image": spaces.Box(low = 0, high = 255, shape = self.image_dim, dtype=np.uint8)})

        self.state_dim = self.observation_space["position"].shape[0]
        self._max_episode_steps = 60 * self.rate 
        self.prev_dyn_state = None
        self.reward_fn_type = reward_fn_type
        self.use_acc_penalty = acc_penalty
        self.fail_penalty = 20
        self.success_reward = 20
        
        ### Loading up Cached Initial Conditions ###
        self.all_ics = []
        with open("./examples/Cassie/cassie_initial_conditions.csv") as csvfile:
            reader = csv.reader(csvfile, delimiter = ",")
            for row in reader:
                self.all_ics.append([float(num) for num in row])
        self.all_ics = np.array(self.all_ics)

        ### Spawning Director Process ###
        if self.viz:
            self.drake_director = sp.Popen(["bazel-bin/director/drake-director", "--use_builtin_scripts=frame,image", "--script", "examples/Cassie/director_scripts/show_time.py"])
            time.sleep(5)  # have to sleep here otherwise visualization throws an error since director has nontrivial startup time 
        else:
            self.drake_director = None
            time.sleep(1)


    # receives and handles the robot state
    def state_handler(self, channel, data):
        msg = lcmt_robot_output.decode(data)
        self.sim_ticks = msg.utime  # potential thread safety issues here.

        # com angle, trans. pos, ang vel, trans. vel
        state = np.array(list(msg.position[0:7]) + list(msg.velocity[0:6]))
        # all_states.append(list(msg.position))
        self.state_queue.put(state)

    # Handles the input image and puts it in queue
    # currently only handles depth images
    def image_handler(self, channel, data):
        msg = lcmt_image_array.decode(data)
        image_msg = msg.images[0]
        self.image_dim = (image_msg.height, image_msg.width, 1)
        image_data = image_msg.data
        if image_msg.bigendian:
            image = Image.frombytes("I;16B", (image_msg.width, image_msg.height), image_data)
        else:
            image = Image.frombytes("I;16", (image_msg.width, image_msg.height), image_data)
        image = np.array(image)
        # since the depth image has values that correspond to real quantities, this is not great.
        # all_images.append(image)
        # big bug--stable baselines 3 expects images to be between 0 and 255!
        # image = image.astype(np.float32)/1000
        image = image/2**8
        image = image.astype(np.uint8)
        # image = image/2**16
        self.image_queue.put(np.reshape(image, self.image_dim))


    def lcm_listener(self):
        while True:
            self.lc.handle() 
            if self.stop_listener.is_set():
                break
            time.sleep(0.001)


    def set_goal(self, x_des, y_des):
        self.goal_state = [x_des, y_des]


    def reward_fn_dist(self, dyn_state):
        x_loc = dyn_state[self.pos_names.index("base_x")]
        y_loc = dyn_state[self.pos_names.index("base_y")]
        return -np.sqrt((x_loc - self.goal_state[0])**2 + (y_loc - self.goal_state[1])**2)


    def reward_fn_firstorder(self, cur_state, prev_state):
        x_loc = prev_state[self.pos_names.index("base_x")]
        y_loc = prev_state[self.pos_names.index("base_y")]
        d2goal_prev = np.sqrt((x_loc - self.goal_state[0])**2 + (y_loc - self.goal_state[1])**2)

        x_loc = cur_state[self.pos_names.index("base_x")]
        y_loc = cur_state[self.pos_names.index("base_y")]
        d2goal_cur = np.sqrt((x_loc - self.goal_state[0])**2 + (y_loc - self.goal_state[1])**2)
        acc_penalty = 0.2 * (np.linalg.norm(cur_state[7:] - prev_state[7:]))
        if self.use_acc_penalty:
            return d2goal_prev - d2goal_cur - acc_penalty
        else:
            return d2goal_prev - d2goal_cur


    def reward_fn_sparse(self, cur_state, prev_state):
        if self.succeeded(cur_state):
            return 100
        else:
            # if current yaw is different than xy velocity vector, penalize it.
            yaw = R.from_quat(cur_state[0:4]).as_euler("zyx")[0]
            yaw_vec = [np.cos(yaw), np.sin(yaw)]
            x_loc = cur_state[self.pos_names.index("base_x")]
            y_loc = cur_state[self.pos_names.index("base_y")]
            # higher reward for facing the goal, with a time penalty
            vec2goal = [self.goal_state[0] - x_loc, self.goal_state[1] - y_loc]
            rew = np.dot(yaw_vec, vec2goal/np.linalg.norm(vec2goal)) - 1
            return rew


    def failed(self, dyn_state):
        x_loc = dyn_state[self.pos_names.index("base_x")]
        y_loc = dyn_state[self.pos_names.index("base_y")]
        z_loc = dyn_state[self.pos_names.index("base_z")]
        
        x_cond = x_loc < self.workspace[0][0] or x_loc > self.workspace[0][1]
        y_cond = y_loc < self.workspace[1][0] or y_loc > self.workspace[1][1]
        z_cond = z_loc < self.workspace[2][0] or z_loc > self.workspace[2][1]
        if x_cond or y_cond or z_cond:
            print("Out of workspace!")
            print("FINAL LOCATION:(", x_loc, y_loc, z_loc, ")")
            return True
        if self.ep_timesteps > self._max_episode_steps:
            print("max episode timesteps exceeded!")
            print("FINAL LOCATION:(", x_loc, y_loc, ")")
            return True
        return False


    # failure if we exit the workspace or success if we hit the goal
    def succeeded(self, dyn_state):
        x_loc = dyn_state[self.pos_names.index("base_x")]
        y_loc = dyn_state[self.pos_names.index("base_y")]
        z_loc = dyn_state[self.pos_names.index("base_z")]
        
        dist_to_goal = np.sqrt((x_loc - self.goal_state[0])**2 + (y_loc - self.goal_state[1])**2)
        if dist_to_goal < 3e-1:
            print("Hit goal!")
            return True
        return False


    def done_fn(self, dyn_state):
        if self.ctrlr is None or self.sim is None:
            return True
        return self.succeeded(dyn_state) or self.failed(dyn_state)


    def normalize_x(self, var):
        return (var - self.workspace[0][0])/(self.workspace[0][0] - self.workspace[0][1])

    def normalize_y(self, var):
        return (var - self.workspace[1][0])/(self.workspace[1][0] - self.workspace[1][1])

    def step(self, action):
        # print(f"action: {action}")
        action_msg = lcmt_radio_out()
        action_msg.channel[0:4] = action

        # send LCM message with the desired action
        self.lc.publish(self.action_channel, action_msg.encode())
        # print("published action", action)
        # the "step" message will step the simulator until the desired simulator time.
        step_msg = lcmt_rl_step()
        step_msg.utime = int(self.sim_ticks + 1/self.rate * 1e6)
        self.lc.publish("LEARNER_STEP_SIM", step_msg.encode())

        # wait and get the last LCM message with the desired robot state
        # have to do this because the sim is running in real time
        # I need a way of only sending an lcm message when the sim is done stepping!
        # time.sleep(1/self.rate)


        # get the next LCM message in the queue
        start = time.time_ns()
        dyn_state = self.state_queue.get(block=True)
        end = time.time_ns()
        # print(f"time to get state (s):{(end - start)/10**9}")
        image = self.image_queue.get(block=True)

        tmp_quat = np.concatenate([dyn_state[1:4], [dyn_state[0]]])
        base_orientation = R.from_quat(tmp_quat).as_euler("zyx")
        base_orientation /= np.pi
        state_reduced = np.array([self.normalize_x(dyn_state[self.pos_names.index("base_x")]),
                                  self.normalize_y(dyn_state[self.pos_names.index("base_y")]),
                                  self.normalize_x(self.goal_state[0]),
                                  self.normalize_y(self.goal_state[1]),
                                  base_orientation[0],
                                  self.ep_timesteps/self._max_episode_steps])
 
        if self.reward_fn_type == 1:
            reward = self.reward_fn_firstorder(dyn_state, self.prev_dyn_state)
            if self.failed(dyn_state):
                reward -= self.fail_penalty
            elif self.succeeded(dyn_state):
                # reward quickly making it to the goal.
                # needs to encounter the goal enough to learn this.
                reward += self.success_reward + (self._max_episode_steps - self.ep_timesteps)/self.rate

        elif self.reward_fn_type == 0:
            reward = self.reward_fn_dist(dyn_state)
        elif self.reward_fn_type == 2:
            reward = self.reward_fn_sparse(dyn_state, self.prev_dyn_state)
            # print(reward)

        self.state = {"position":state_reduced, "image":image}
        self.done = self.done_fn(dyn_state)
        self.ep_timesteps += 1
        if self.done:
            self.kill_procs()
        self.prev_dyn_state = dyn_state
        return self.state, reward, self.done, {}


    def reset(self):
        # kill the simulation, reset the controller, stop the listener thread
        self.kill_procs()

        # clear the queues (for now just making new queues)
        self.state_queue = queue.LifoQueue()
        self.image_queue = queue.LifoQueue()

        # self.set_goal(np.random.uniform(self.workspace[0][0], self.workspace[0][1]),
        #              np.random.uniform(self.workspace[1][0], self.workspace[1][1]))
        # self.set_goal(self.goal_state[0], self.goal_state[1])

        # pick a new initial condition, set that as our state
        ic_idx = np.random.randint(0, self.all_ics.shape[1])
        # print("using ic", ic_idx)
        ic = self.all_ics[:,ic_idx]
        
        self.ctrlr = sp.Popen([self.bin_dir + self.controller_p] + self.ctrlr_options)
        t_string = "--terrain_type=" + str(self.terrain_class)
        self.sim = sp.Popen([self.bin_dir + self.simulation_p,
                            "--ic_idx=" + str(ic_idx),
                            "--num_obstacles="+str(self.num_features),
                            "--viz="+str(int(self.viz)), 
                            "--target_realtime_rate=0.01", t_string,
                            "--dt="+str(0.00009)])
        time.sleep(1.5)
        # vary the goal state
        # random_ang = np.random.rand() * np.pi
        # self.set_goal(np.cos(random_ang)*self.goal_dist, np.sin(random_ang)*self.goal_dist)
        print(f"set goal to:{self.goal_state}")

        if self.stop_listener.is_set():
            self.stop_listener.clear()

        self.listener_thread = threading.Thread(target=self.lcm_listener)
        self.listener_thread.start()

        # start the lcm driven loop
        self.sim_ticks = 0e6
        step_msg = lcmt_rl_step()
        step_msg.utime = int(0)
        self.lc.publish("LEARNER_STEP_SIM", step_msg.encode())

        '''
        self.sim_ticks = 1e6  # force it to initialize
        step_msg = lcmt_rl_step()
        step_msg.utime = int(self.sim_ticks + 1/self.rate * 1e6)
        self.lc.publish("LEARNER_STEP_SIM", step_msg.encode())
        '''

        self.ep_timesteps = 0
        dyn_state = self.state_queue.get(block=True)
        self.prev_dyn_state = dyn_state
        image = self.image_queue.get(block=True)
        base_orientation = R.from_quat(dyn_state[0:4]).as_euler("zyx")
        state_reduced = np.array([self.normalize_x(dyn_state[self.pos_names.index("base_x")]),
                                  self.normalize_y(dyn_state[self.pos_names.index("base_y")]),
                                  self.normalize_x(self.goal_state[0]),
                                  self.normalize_y(self.goal_state[1]),
                                  base_orientation[0]/np.pi,
                                  self.ep_timesteps/self._max_episode_steps])
        self.state = {"position":state_reduced, "image":image}
        return self.state


    def kill_procs(self):
        if self.sim is not None:
            self.sim.terminate()
            self.sim = None
        if self.ctrlr is not None:
            self.ctrlr.terminate()
            self.ctrlr = None
        if not self.stop_listener.is_set():
            self.stop_listener.set()
        
    def kill_director(self):
        if self.drake_director is not None:
            self.drake_director.terminate()


class Cassie_FixedInit(CassieEnv_Joystick):
    def __init__(self,
                action_channel,
                state_channel,
                rate,
                workspace,
                goal_state,
                visualize,
                terrain_class,
                num_features):
        super().__init__(action_channel, state_channel, rate, workspace, goal_state, visualize,
                         terrain_class, num_features, reward_fn_type = 1, acc_penalty = False)

        self.fail_penalty = 20
        self.success_reward = 20
        self.goal_max = goal_state[0]


    def reset(self):
        # kill the simulation, reset the controller, stop the listener thread
        self.kill_procs()

        # clear the queues (for now just making new queues)
        print(f"queue size on reset:{self.state_queue.qsize()}")
        self.state_queue = queue.LifoQueue()
        self.image_queue = queue.LifoQueue()


        # self.goal_state = [np.random.uniform(1, self.goal_max), 0]

        # pick a new initial condition, set that as our state
        ic_idx = 100
        ic = self.all_ics[:,ic_idx]
        
        self.ctrlr = sp.Popen([self.bin_dir + self.controller_p] + self.ctrlr_options)
        t_string = "--terrain_type=" + str(self.terrain_class)
        # 9e-5 instead of 8e-5 gives a slight speedup in sim time, though not that much.
        self.sim = sp.Popen([self.bin_dir + self.simulation_p,
                            "--ic_idx=" + str(ic_idx),
                            "--num_obstacles="+str(self.num_features),
                            "--viz="+str(int(self.viz)), 
                            "--target_realtime_rate=0.0", t_string,"--dt="+str(0.00009)])
        time.sleep(1.5)

        if self.stop_listener.is_set():
            self.stop_listener.clear()

        self.listener_thread = threading.Thread(target=self.lcm_listener)
        self.listener_thread.start()

        # start the lcm driven loop
        self.sim_ticks = 0e6
        step_msg = lcmt_rl_step()
        step_msg.utime = int(0)
        self.lc.publish("LEARNER_STEP_SIM", step_msg.encode())

        self.ep_timesteps = 0
        dyn_state = self.state_queue.get(block=True)
        self.prev_dyn_state = dyn_state
        image = self.image_queue.get(block=True)

        base_orientation = R.from_quat(dyn_state[0:4]).as_euler("zyx")
        state_reduced = np.array([self.normalize_x(dyn_state[self.pos_names.index("base_x")]),
                                  self.normalize_y(dyn_state[self.pos_names.index("base_y")]),
                                  self.normalize_x(self.goal_state[0]),
                                  self.normalize_y(self.goal_state[1]),
                                  base_orientation[0],
                                  self.ep_timesteps/self._max_episode_steps])
        self.state = {"position":state_reduced, "image":image}
        return self.state

    
# Wrapper class that removes perception
class Cassie_FixedGoal_Blind(Cassie_FixedInit):
    def __init__(self,
                action_channel,
                state_channel,
                rate,
                workspace,
                goal_state,
                visualize):
        super().__init__(action_channel, state_channel, rate, workspace, goal_state, visualize, terrain_class = 2, num_features = 0)

        self.observation_space = spaces.Box(low = np.array([0, 0, 0, 0, 0, 0]), high = np.array([1, 1, 1, 1, 1, 1]))
        self.state_dim = self.observation_space.shape[0]

    def step(self, action):
        s, r, d, info = super().step(action)
        self.state = s["position"]
        return self.state, r, d, info

    def reset(self):
        state = super().reset()
        self.state = state["position"]
        return self.state


# Wrapper class that removes perception
class Cassie_RandGoal_Blind(Cassie_FixedInit):
    def __init__(self,
                action_channel,
                state_channel,
                rate,
                workspace,
                visualize):
        super().__init__(action_channel, state_channel, rate, workspace, [0, 0], visualize, terrain_class = 2, num_features = 0)

        self.observation_space = spaces.Box(low = np.array([0, 0, 0, 0, 0, 0]), high = np.array([1, 1, 1, 1, 1, 1]))
        self.state_dim = self.observation_space.shape[0]

    def step(self, action):
        s, r, d, info = super().step(action)
        self.state = s["position"]
        return self.state, r, d, info

    def reset(self):
        state = super().reset()
        # set a random goal in the workspace
        self.set_goal(np.random.uniform(self.workspace[0][0], self.workspace[0][1]),
                      np.random.uniform(self.workspace[1][0], self.workspace[1][1]))
        print(f"NEW GOAL STATE: {self.goal_state}")
        self.state = state["position"]
        return self.state


class Cassie_RandGoalObst_Blind(Cassie_FixedInit):
    def __init__(self,
                action_channel,
                state_channel,
                rate,
                workspace,
                visualize):
        super().__init__(action_channel, state_channel, rate, workspace, [0, 0], visualize, terrain_class = 4, num_features = 0)

        self.observation_space = spaces.Box(low = np.array([0, 0, 0, 0, 0, 0]), high = np.array([1, 1, 1, 1, 1, 1]))
        self.state_dim = self.observation_space.shape[0]

    def step(self, action):
        s, r, d, info = super().step(action)
        self.state = s["position"]
        return self.state, r, d, info

    def reset(self):
        state = super().reset()
        # set a random goal in the workspace
        self.set_goal(np.random.uniform(self.workspace[0][0], self.workspace[0][1]),
                      np.random.uniform(self.workspace[1][0], self.workspace[1][1]))
        print(f"NEW GOAL STATE: {self.goal_state}")
        self.state = state["position"]
        return self.state


class Cassie_FixedGoal_Depth(Cassie_FixedInit):
    def __init__(self,
                action_channel,
                state_channel,
                rate,
                workspace,
                goal_state,
                visualize,
                ditches = False):
        if not ditches:
            super().__init__(action_channel, state_channel, rate, workspace, goal_state, visualize,
                             terrain_class = 5, num_features = 0)
        else:
            super().__init__(action_channel, state_channel, rate, workspace, goal_state, visualize,
                             terrain_class = 6, num_features = 0)
        

        self.fail_penalty = 20

        # try making this 0 to encourage more exploration
        # self.success_reward = 20
        self.success_reward = 0



def main():
    try: 
        workspace = [[-1, 5], [-3, 3], [0.5, 1.3]]
        env = Cassie_FixedGoal_Depth("CASSIE_VIRTUAL_RADIO", "CASSIE_STATE_RL", 2, workspace, [1, 0], True, False)
        # env = CassieEnv_Joystick("CASSIE_VIRTUAL_RADIO", "CASSIE_STATE_RL", 2, workspace, [4, 4], True, 4, 0, 1, False)
        s = env.reset()
        i = 0
        while i < 20: 
            s, r, d, _ = env.step([0.2, 0.0, 0.0, 0.0])
            time.sleep(1)
            if d:
                break
            i += 1
        while i < 30: 
            s, r, d, _ = env.step([0.3, 0.0, 0.0, 0.0])
            time.sleep(0.5)
            if d:
                break
            i += 1

        env.kill_procs()
        env.kill_director()
    except KeyboardInterrupt:
        env.kill_procs()
        env.kill_director()

if __name__ == "__main__":
    main()
