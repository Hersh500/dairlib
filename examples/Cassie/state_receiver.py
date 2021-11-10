import lcm
from dairlib import lcmt_robot_output, lcmt_image_array
import subprocess as sp

bin_dir = "./bazel-bin/examples/Cassie/"
controller_p = "run_osc_standing_controller" 
simulation_p = "multibody_sim"

counter = 1
def handler(channel, data):
    global counter
    counter += 1
    if counter % 10 == 0:
        # msg = lcmt_robot_output.decode(data)
        msg = lcmt_image_array.decode(data)
        print("Received message!")
        print("num_images = ", str(msg.num_images))
        counter = 1

lc = lcm.LCM()
sub = lc.subscribe("DRAKE_RGBD_CAMERA_IMAGES", handler)

# ctrlr = sp.Popen([bin_dir + controller_p, "--height=0.9"])
# sim = sp.Popen([bin_dir + simulation_p, "--init-height=0.9"])

try:
    while True:
        lc.handle()
except KeyboardInterrupt:
    ctrlr.terminate()
    sim.terminate()
    pass
