# https://lcm-proj.github.io/tut_python.html
import lcm
from dairlib import lcmt_robot_output, lcmt_pd_config
import subprocess as sp
import gym
import queue

# Assuming running from dairlib/
bin_dir = "./bazel-bin/examples/Cassie/"
controller_p = "run_osc_standing_controller" 
simulation_p = "multibody_sim"

counter = 1
def handler(channel, data):
    global counter
    counter += 1
    if counter % 1000 == 0:
        msg = lcmt_robot_output.decode(data)
        print("Received message!")
        print("timestamp = %s" % str(msg.position_names))
        counter = 1

bin_dir = "/home/hersh/Programming/dairlib/bazel-bin/examples/Cassie/"
controller_p = "run_osc_standing_controller" 
simulation_p = "multibody_sim"

def main():
    ctrlr = sp.Popen([bin_dir + controller_p, "--height=0.9"])
    sim = sp.Popen([bin_dir + simulation_p, "--init-height=0.9"])

    try:
        while True:
            lc.handle()
    except KeyboardInterrupt:
        ctrlr.terminate()
        sim.terminate()
        pass

if __name__ == "__main__":
    main()

class CassieEnv_test(gym.Env):
    def __init__(self, action_channel, state_channel, rate):
        super(CassieEnv, self).__init__()
        # precompute random initial conditions for domain randomization

        # set the terrain (TODO: is it fully randomized every time?)
        # spawn the controller, and keep track of the pid
        self.ctrlr = sp.Popen([bin_dir + controller_p, "--height=0.9"])
        # spawn the simulation, and keep track of the pid (to kill to reset the sim)
        self.sim = None

        lc = lcm.LCM()
        sub = lc.subscribe(state_channel, state_handler)
        self.state_queue = queue.LifoQueue()
        self.action_channel = action_channel
        self.rate = rate

        # receives and handles the robot state
        def state_handler(channel, data):
            msg = lcmt_robot_output.decode(data)
            print("Received message!")
            print("timestamp = %s" % str(msg.position_names))
            self.state_queue.put(msg_decoded)

        self.done = False
        return


    # Need to have some nominal gains for PD
    def step(self, action):
        # see examples/director_scripts/pd_panel.py
        action_msg = lcmt_pd_config()
        # massage the action into the desired output message form

        # send LCM message with the desired action
        self.lcm.publish(self.action_channel, action_msg.encode())
        # wait and get the last LCM message with the desired robot state
        time.sleep(1/self.rate)
        # get the next LCM message in the queue
        self.state = self.state_queue.get(block=True)
        # check based on self.state
        if some_bad_condition:
            self.done = True

        # compute the reward from the state
        return self.state, reward, self.done


    def reset(self, terrain_des = None):
        # clear the state queue
        # kill the simulation (reset the controller)
        if self.sim is not None:
            self.sim.terminate()
        # pick a new initial condition, set that as our state
        self.sim = sp.Popen([bin_dir + simulation_p, "--init-height=0.9"])
        # can pass in terrain_des if you want to evaluate
        # wait until we receive a state from the simulation to start doing stuff.
        print("resetting and waiting for new state...")
        self.state = self.state_queue.get(block=True)
        return self.state


if __name__ == "__main__":
    main()
