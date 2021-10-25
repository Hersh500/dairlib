# https://lcm-proj.github.io/tut_python.html
import lcm
from dairlib import lcmt_robot_output, lcmt_pd_config
import subprocess as sp
import gym
import queue
import numpy as np
import time
import csv

    
class CassieEnv_test(gym.Env):
    def __init__(self, action_channel, state_channel, rate):
        super(CassieEnv_test, self).__init__()

        # spawn the controller, and keep track of the pid
        self.ctrlr = None
        # spawn the simulation, and keep track of the pid (to kill to reset the sim)
        self.sim = None

        self.lc = lcm.LCM()
        self.sub = self.lc.subscribe(state_channel, self.state_handler)
        self.state_queue = queue.LifoQueue()
        self.action_channel = action_channel
        self.state_channel = state_channel
        self.rate = rate

        self.done = False
        # for now just fix these...
        # are these gains fucked? robot keeps falling over...
        self.joint_default = [-0.01,.01,0,0,0.55,0.55,-1.5,-1.5,-1.8,-1.8]
        self.kp_default = [i for i in [80,80,50,50,50,50,50,50,10,10]]
        self.kd_default = [i for i in [1,1,1,1,1,1,2,2,1,1]]

        self.height_des = 0.8
        self.pos_names = ['base_qw', 'base_qx', 'base_qy', 'base_qz', 'base_x', 'base_y', 'base_z', 'hip_roll_left', 'hip_roll_right', 'hip_yaw_left', 'hip_yaw_right', 'hip_pitch_left', 'hip_pitch_right', 'knee_left', 'knee_right', 'knee_joint_left', 'knee_joint_right', 'ankle_joint_left', 'ankle_joint_right', 'ankle_spring_joint_left', 'toe_left', 'ankle_spring_joint_right', 'toe_right']

        self.joint_names = [
            "hip_roll_left_motor",
            "hip_roll_right_motor",
            "hip_yaw_left_motor",
            "hip_yaw_right_motor",
            "hip_pitch_left_motor",
            "hip_pitch_right_motor",
            "knee_left_motor",
            "knee_right_motor",
            "toe_left_motor",
            "toe_right_motor"]

        self.joints_in_pos_array = [7, 8, 9, 10, 11, 12, 13, 14, 20, 22]

        # Assuming running from dairlib/
        self.bin_dir = "./bazel-bin/examples/Cassie/"
        self.controller_p = "run_pd_controller"
        # self.controller_p = "run_osc_standing_controller"
        self.simulation_p = "rl_multibody_sim"

        # get grid of all initial conditions
        self.all_ics = []
        with open("./examples/Cassie/cassie_initial_conditions.csv") as csvfile:
            reader = csv.reader(csvfile, delimiter = ",")
            for row in reader:
                self.all_ics.append([float(num) for num in row])
        self.all_ics = np.array(self.all_ics)
        return


    # receives and handles the robot state
    def state_handler(self, channel, data):
        msg = lcmt_robot_output.decode(data)
        state = list(msg.position) + list(msg.velocity)
        self.state_queue.put(state)


    def step(self, action):
        # see examples/director_scripts/pd_panel.py
        action_msg = lcmt_pd_config()

        # massage the action into the desired output message form
        action_msg.num_joints = len(self.joint_default)
        action_msg.joint_names = self.joint_names
        action_msg.desired_velocity = [0] * action_msg.num_joints
        action_msg.desired_position = action
        action_msg.timestamp = int(time.time() * 1e6)
        action_msg.kp = self.kp_default
        action_msg.kd = self.kd_default

        # send LCM message with the desired action
        self.lcm.publish(self.action_channel, action_msg.encode())

        # wait and get the last LCM message with the desired robot state
        time.sleep(1/self.rate)

        while self.state_queue.qsize() < 1:
            self.lc.handle()

        # get the next LCM message in the queue
        self.state = self.state_queue.get(block=True)

        # check on self.state (maybe say failure is CoM height below a threshold = falling?)
        if self.state[self.joint_map["pelvis_z"]] < 0.4:
            self.done = True
            self.sim.terminate()
            self.sim = None
            reward = 0
        else:
            self.done = False
            # really simple reward for now
            reward = -np.abs(self.height_des - self.state[self.pos_names.index("pelvis_z")])

        return self.state, reward, self.done


    def reset(self, terrain_des = None):
        # clear the state queue
        # kill the simulation (reset the controller)
        self.kill_procs()

        # pick a new initial condition, set that as our state
        ic_idx = np.random.randint(0, self.all_ics.shape[1])
        print("using ic", ic_idx)
        ic = self.all_ics[:,ic_idx]
        
        # init_height = np.random.rand() * 0.3 + 0.6  # range of [0.6, 0.9)
        self.ctrlr = sp.Popen([self.bin_dir + self.controller_p, "--channel_x=" + self.state_channel])
        self.sim = sp.Popen([self.bin_dir + self.simulation_p, "--ic_idx=" + str(ic_idx)])
        # self.sim = sp.Popen([self.bin_dir + self.simulation_p])

        # TODO: do I need to send a nominal action message so the robot doesn't fall over?
        # wait until we receive a state from the simulation to start doing stuff.
        while self.state_queue.qsize() < 1:
            self.lc.handle()
        self.state = self.state_queue.get(block=True)

        print("publishing nominal joint state")
        action_msg = lcmt_pd_config()
        action_msg.num_joints = len(self.joint_default)
        action_msg.joint_names = self.joint_names
        action_msg.desired_velocity = [0] * action_msg.num_joints
        action_msg.desired_position = ic[self.joints_in_pos_array]
        action_msg.timestamp = int(time.time() * 1e6)
        action_msg.kp = self.kp_default
        action_msg.kd = self.kd_default
        self.lc.publish(self.action_channel, action_msg.encode())
        
        return self.state


    def kill_procs(self):
        if self.sim is not None:
            self.sim.terminate()
        if self.ctrlr is not None:
            self.ctrlr.terminate()
        

def main():
    env = CassieEnv_test("PD_CONFIG", "CASSIE_STATE_SIMULATION", 200)
    s = env.reset()
    time.sleep(10)
    env.kill_procs()

if __name__ == "__main__":
    main()
