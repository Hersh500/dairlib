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

lc = lcm.LCM()
sub = lc.subscribe("CASSIE_STATE_SIMULATION", handler)

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


class CassieEnv_test(gym.Env):
    def __init__(self, action_channel, state_channel, rate):
        super(CassieEnv, self).__init__()
        # precompute random initial conditions for domain randomization

        # set the terrain (TODO: is it fully randomized every time?)
        # spawn the controller, and keep track of the pid
        self.ctrlr = sp.Popen([bin_dir + controller_p, "--channel_x=CASSIE_STATE_SIMULATION"])
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
            state = list(msg.position) + list(msg.velocity)
            self.state_queue.put(state)

        self.done = False

        # for now just fix these...
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

        return


    # Need to have some nominal gains for PD
    def step(self, action):
        # see examples/director_scripts/pd_panel.py
        action_msg = lcmt_pd_config()

        # massage the action into the desired output message form
        action_msg.num_joints = len(self.joint_default)
        action_msg.joint_names = self.joint_names
        action_msg.desired_velocity = [0] * action_msg.num_joints
        action_msg.desired_position = action
        action_msg.timestamp = int(time.time() * 1e6)
        action_msg.kp = self.default_kp
        action_msg.kp = self.default_kd

        # send LCM message with the desired action
        self.lcm.publish(self.action_channel, action_msg.encode())

        # wait and get the last LCM message with the desired robot state
        time.sleep(1/self.rate)

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
        if self.sim is not None:
            self.sim.terminate()
        # pick a new initial condition, set that as our state
        init_height = np.random.rand() * 0.4 + 0.5  # range of [0.5, 0.9)
        self.sim = sp.Popen([bin_dir + simulation_p, "--init-height=" + str(init_height)])

        # wait until we receive a state from the simulation to start doing stuff.
        print("resetting and waiting for new state...")
        self.state = self.state_queue.get(block=True)
        return self.state


if __name__ == "__main__":
    main()
