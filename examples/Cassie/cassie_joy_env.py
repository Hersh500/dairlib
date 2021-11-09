# https://lcm-proj.github.io/tut_python.html
import lcm
from dairlib import lcmt_robot_output, lcmt_radio_out
import subprocess as sp
import gym
import queue
import numpy as np
import time
import csv

    
# TODO(hersh): make a more flexible environment so it's easier to slot in other
# low level controllers and tasks
# requires: defining your own action -> message, message -> state, state -> reward,
# lcm message types, termination conditions... seems to be pretty annoying anyways.
# maybe it's just a more structured way to go about things?
class CassieEnv_Joystick(gym.Env):
    def __init__(self, action_channel, state_channel, rate):
        super(CassieEnv_Joystick, self).__init__()

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

        # Assuming running from dairlib/
        self.bin_dir = "./bazel-bin/examples/Cassie/"
        self.controller_p = "run_osc_walking_controller"
        self.simulation_p = "rl_multibody_sim"
        self.ctrlr_options = ["--use_radio=1", "--cassie_out_channel=CASSIE_OUTPUT", "--channel_x="+self.state_channel]

        # get grid of all initial conditions
        self.all_ics = []
        with open("./examples/Cassie/cassie_initial_conditions.csv") as csvfile:
            reader = csv.reader(csvfile, delimiter = ",")
            for row in reader:
                self.all_ics.append([float(num) for num in row])
        self.all_ics = np.array(self.all_ics)

        # spawn the director and visualizer
        self.drake_director = sp.Popen(["bazel-bin/director/drake-director", "--use_builtin_scripts=frame,image", "--script", "examples/Cassie/director_scripts/show_time.py"])
        # have to sleep here otherwise visualization throws an error since director takes time to startup
        time.sleep(5)
        return


    # receives and handles the robot state
    def state_handler(self, channel, data):
        msg = lcmt_robot_output.decode(data)
        # com angle, trans. pos, ang vel, trans. vel
        state = np.array(list(msg.position[0:7]) + list(msg.velocity[0:6]))
        self.state_queue.put(state)

    # TODO: need to get the drake image output type somehow
    def step(self, action, goal_state):
        # see examples/director_scripts/pd_panel.py
        action_msg = lcmt_radio_out()
        action_msg.channel[0:4] = action  # does slicing work here?

        # send LCM message with the desired action
        self.lc.publish(self.action_channel, action_msg.encode())

        # wait and get the last LCM message with the desired robot state
        time.sleep(1/self.rate)

        while self.state_queue.qsize() < 1:
            self.lc.handle()

        # get the next LCM message in the queue
        self.state = self.state_queue.get(block=True)

        # check on self.state (maybe say failure is CoM height below a threshold = falling?)
        '''
        if self.state[self.joint_map["pelvis_z"]] < 0.4:
            self.done = True
            self.sim.terminate()
            self.sim = None
            reward = 0
        else:
            self.done = False
            # really simple reward for now
        '''
        # TODO: dimensionality reduction on the state to get the COM distance to the goal, COM velocity

        reward = -np.abs(self.state - goal_state) * 0

        return self.state, reward, self.done


    def reset(self, terrain_des = None):
        # clear the state queue
        # kill the simulation (reset the controller)
        self.kill_procs()

        # pick a new initial condition, set that as our state
        ic_idx = np.random.randint(0, self.all_ics.shape[1])
        print("using ic", ic_idx)
        ic = self.all_ics[:,ic_idx]
        
        self.ctrlr = sp.Popen([self.bin_dir + self.controller_p] + self.ctrlr_options)
        self.sim = sp.Popen([self.bin_dir + self.simulation_p, "--ic_idx=" + str(ic_idx)])

        # TODO: do I need to send a nominal action message so the robot doesn't fall over?
        # wait until we receive a state from the simulation to start doing stuff.
        while self.state_queue.qsize() < 1:
            self.lc.handle()
        self.state = self.state_queue.get(block=True)
        return self.state


    def kill_procs(self):
        if self.sim is not None:
            self.sim.terminate()
        if self.ctrlr is not None:
            self.ctrlr.terminate()
        
    def kill_director(self):
        self.drake_director.terminate()

def main():
    try: 
        env = CassieEnv_Joystick("CASSIE_VIRTUAL_RADIO", "CASSIE_STATE_SIMULATION", 200)
        s = env.reset()
        s, r, d = env.step([0.0, 0, 0, 0], 0)  # just to see what happens
        time.sleep(90)
        env.kill_procs()
        env.kill_director()
    except KeyboardInterrupt:
        env.kill_procs()
        env.kill_director()

if __name__ == "__main__":
    main()
