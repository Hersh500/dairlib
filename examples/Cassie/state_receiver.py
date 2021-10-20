# https://lcm-proj.github.io/tut_python.html
import lcm
from dairlib import lcmt_robot_output
import subprocess as sp
import gym

bin_dir = "/home/hersh/Programming/dairlib/bazel-bin/examples/Cassie/"
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
    def __init__(self):
        super(CassieEnv, self).__init__()
        # precompute random initial conditions for domain randomization

        # set the terrain (TODO: is it fully randomized every time?)
        # spawn the controller, and keep track of the pid
        # spawn the simulation, and keep track of the pid (to kill to reset the sim)
        return


    def step(self, action):
        # send LCM message with the desired action (TODO: where should this be ZOH'd?)
        # wait and get the last LCM message with the desired robot state 
        return

    def reset(self, terrain_des = None):
        # kill the simulation (reset the controller)
        # restart the simulation and pick a new initial condition
        # can pass in terrain_des if you want to evaluate
        return
