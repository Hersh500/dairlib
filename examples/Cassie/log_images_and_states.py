# This is an abomination. I should instead have some more modular files or something.
import lcm
from dairlib import lcmt_robot_output, lcmt_image_array, lcmt_image, lcmt_rl_step
from PIL import Image

states = []
images = []

def state_handler(channel, data):
    msg = lcmt_robot_output.decode(data)

    # com angle, trans. pos, ang vel, trans. vel
    state = np.array(list(msg.position[0:7]) + list(msg.velocity[0:6]))
    states.append(list(msg.position))

# Handles the input image and puts it in queue
# currently only handles depth images
def image_handler(channel, data):
    msg = lcmt_image_array.decode(data)
    image_msg = msg.images[0]
    image_dim = (image_msg.height, image_msg.width)
    image_data = image_msg.data
    if image_msg.bigendian:
        image = Image.frombytes("I;16B", (image_msg.width, image_msg.height), image_data)
    else:
        image = Image.frombytes("I;16", (image_msg.width, image_msg.height), image_data)
    image = np.array(image)
    images.append(image)


# spawn the processes
bin_dir = "./bazel-bin/examples/Cassie/"
controller_p = "run_osc_walking_controller"
simulation_p = "rl_multibody_sim"
ctrlr_options = ["--use_radio=1", "--cassie_out_channel=CASSIE_OUTPUT", "--channel_x="+state_channel]


action_channel = "CASSIE_VIRTUAL_RADIO"
state_channel = "CASSIE_STATE_SIMULATION"

lc = lcm.LCM()
sub_state = lc.subscribe(state_channel, state_handler)
sub_images = lc.subscribe("DRAKE_RGBD_CAMERA_IMAGES", image_handler)

ctrlr = sp.Popen([bin_dir + controller_p] + ctrlr_options)
sim_gaps = sp.Popen([bin_dir + simulation_p, "--ic_idx=" + str(ic_idx), "--num_obstacles="+str(6), "--gaps=1"])
sim_obst = sp.Popen([bin_dir + simulation_p, "--ic_idx=" + str(ic_idx), "--num_obstacles="+str(6), "--gaps=0"])


ticks = 0
step_msg = lcmt_rl_step()
step_msg.utime = int(0)
lc.publish("LEARNER_STEP_SIM", step_msg.encode())

while i < 40: 
    action_msg = lcmt_radio_out()
    action_msg.channel[0:4] = [0, 0, 0, 0] 

    # send LCM message with the desired action
    lc.publish(self.action_channel, action_msg.encode())

    step_msg = lcmt_rl_step()
    step_msg.utime = int(self.sim_ticks + 1/self.rate * 1e6)
    self.lc.publish("LEARNER_STEP_SIM", step_msg.encode())
    i += 1
    time.sleep(0.05)
