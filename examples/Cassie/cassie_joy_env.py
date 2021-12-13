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

# temporary
all_images = []    
all_states = []

# TODO(hersh): make a more flexible environment so it's easier to slot in other
# low level controllers and tasks (like footstep placement)
# requires: defining your own action -> message, message -> state, state -> reward,
# lcm message types, termination conditions... seems to be pretty annoying anyways.
# maybe it's just a more structured way to go about things?
class CassieEnv_Joystick(gym.Env):
    def __init__(self, action_channel, state_channel, rate, workspace, visualize):
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
        self.ctrlr_options = ["--use_radio=1", "--cassie_out_channel=CASSIE_OUTPUT", "--channel_x="+self.state_channel]
        self.sim_ticks = 0

        ### Setting RL variables ###
        self.workspace = workspace
        self.goal_state = [5, 5]
        self.done = False
        self.rate = rate
        self.image_dim = (1, 128, 128)
        self.action_space = spaces.Box(low = np.array([-1, -1, -1, -1]), high = np.array([1, 1, 1, 1]))
        # TODO(hersh): this breaks the current td3 code.
        self.observation_space = spaces.Dict({"position": spaces.Box(low = np.array([-5, -5, -5, -5, -np.pi]), high = np.array([5, 5, 5, 5, np.pi])),
                                              "image": spaces.Box(low = -1, high = 1, shape = self.image_dim)})
        self.state_dim = 5
        self._max_episode_steps = 30 * self.rate 
        self.prev_dyn_state = None
        
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



    # receives and handles the robot state
    def state_handler(self, channel, data):
        msg = lcmt_robot_output.decode(data)
        self.sim_ticks = msg.utime  # potential thread safety issues here.

        # com angle, trans. pos, ang vel, trans. vel
        state = np.array(list(msg.position[0:7]) + list(msg.velocity[0:6]))
        all_states.append(list(msg.position))
        self.state_queue.put(state)

    # Handles the input image and puts it in queue
    # currently only handles depth images
    def image_handler(self, channel, data):
        msg = lcmt_image_array.decode(data)
        image_msg = msg.images[0]
        self.image_dim = (image_msg.height, image_msg.width)
        image_data = image_msg.data
        if image_msg.bigendian:
            image = Image.frombytes("I;16B", (image_msg.width, image_msg.height), image_data)
        else:
            image = Image.frombytes("I;16", (image_msg.width, image_msg.height), image_data)
        image = np.array(image)
        # since the depth image has values that correspond to real quantities, this is not great.
        all_images.append(image)
        image = image.astype(np.float32)
        image = image/2**16
        self.image_queue.put(np.expand_dims(image, axis=0))


    def lcm_listener(self):
        while True:
            self.lc.handle() 
            if self.stop_listener.is_set():
                break
            time.sleep(0.001)


    def set_goal(self, x_des, y_des):
        self.goal_state = [x_des, y_des]


    def reward_fn(self, dyn_state):
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
        return d2goal_prev - d2goal_cur


    # failure if we exit the workspace or success if we hit the goal
    def done_fn(self, dyn_state):
        if self.ctrlr is None or self.sim is None:
            return True
        x_loc = dyn_state[self.pos_names.index("base_x")]
        y_loc = dyn_state[self.pos_names.index("base_y")]
        z_loc = dyn_state[self.pos_names.index("base_z")]
        
        x_cond = x_loc < self.workspace[0][0] or x_loc > self.workspace[0][1]
        y_cond = y_loc < self.workspace[1][0] or y_loc > self.workspace[1][1]
        z_cond = z_loc < self.workspace[2][0] or z_loc > self.workspace[2][1]
        if x_cond or y_cond or z_cond:
            print("Out of workspace!")
            print("FINAL LOCATION:(", x_loc, y_loc, ")")
            return True
        dist_to_goal = np.sqrt((x_loc - self.goal_state[0])**2 + (y_loc - self.goal_state[1])**2)
        if dist_to_goal < 3e-1:
            return True
        if self.ep_timesteps > self._max_episode_steps:
            print("max episode timesteps exceeded!")
            print("FINAL LOCATION:(", x_loc, y_loc, ")")
            return True
        return False


    def step(self, action):
        # see examples/director_scripts/pd_panel.py
        action_msg = lcmt_radio_out()
        action_msg.channel[0:4] = action

        # send LCM message with the desired action
        self.lc.publish(self.action_channel, action_msg.encode())
        # print("published action", action)
        step_msg = lcmt_rl_step()
        step_msg.utime = int(self.sim_ticks + 1/self.rate * 1e6)
        self.lc.publish("LEARNER_STEP_SIM", step_msg.encode())

        # wait and get the last LCM message with the desired robot state
        # have to do this because the sim is running in real time
        # TODO: can this run faster than real time to get more training in?
        time.sleep(1/self.rate)

        '''
        while self.state_queue.qsize() < 1:
            self.lc.handle()
        while self.image_queue.qsize() < 1:
            self.lc.handle()
        '''

        # get the next LCM message in the queue
        dyn_state = self.state_queue.get(block=True)
        image = self.image_queue.get(block=True)

        base_orientation = R.from_quat(dyn_state[0:4]).as_euler("zyx")[0]
        state_reduced = np.array([dyn_state[self.pos_names.index("base_x")],
                                  dyn_state[self.pos_names.index("base_y")],
                                  self.goal_state[0],
                                  self.goal_state[1],
                                  base_orientation])
        reward = self.reward_fn_firstorder(dyn_state, self.prev_dyn_state)
        self.state = {"position":state_reduced, "image":image}
        self.done = self.done_fn(dyn_state)
        self.ep_timesteps += 1
        if self.done:
            self.kill_procs()
        # print("got reward", reward)
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
        self.set_goal(3, 3)

        # pick a new initial condition, set that as our state
#        ic_idx = np.random.randint(0, self.all_ics.shape[1])
        ic_idx = 200
        print("using ic", ic_idx)
        ic = self.all_ics[:,ic_idx]
        
        self.ctrlr = sp.Popen([self.bin_dir + self.controller_p] + self.ctrlr_options)
        self.sim = sp.Popen([self.bin_dir + self.simulation_p, "--ic_idx=" + str(ic_idx), "--num_obstacles="+str(6), "--viz="+str(int(self.viz)), "--gaps=0"])
        time.sleep(1)


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
        base_orientation = R.from_quat(dyn_state[0:4]).as_euler("zyx")[0]
        state_reduced = np.array([dyn_state[self.pos_names.index("base_x")],
                                  dyn_state[self.pos_names.index("base_y")],
                                  self.goal_state[0],
                                  self.goal_state[1],
                                  base_orientation])
        # self.state = (state_reduced, image)
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


def main():
    try: 
        workspace = [[-1, 5], [-3, 3], [0.5, 1.3]]
        env = CassieEnv_Joystick("CASSIE_VIRTUAL_RADIO", "CASSIE_STATE_SIMULATION", 10, workspace, True)
        s = env.reset()
        i = 0
        while i < 1000: 
            s, r, d, _ = env.step([-0.2, 0, 0, 0])  # just to see what happens
            i += 1
            time.sleep(0.05)

        # print('exited step')
        time.sleep(90)
        # TODO: figure out how to detect when the controller terminates
        env.kill_procs()
        env.kill_director()
    except KeyboardInterrupt:
        env.kill_procs()
        env.kill_director()
        # print("saving...")
        # np.save("ref_images", np.array(all_images))
        # np.save("ref_states", np.array(all_states))

if __name__ == "__main__":
    main()
