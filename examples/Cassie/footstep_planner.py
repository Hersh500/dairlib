# Takes a nominal footstep through an lcmt_saved_traj
# and an image message, and then plans a footstep based on that
import lcm
import numpy as np
import argparse
from skimage import filters
import threading
from pytransform3d import transformations as pt
from pytransform3d.rotations import active_matrix_from_extrinsic_roll_pitch_yaw

from dairlib import lcmt_saved_traj, lcmt_robot_out, lcmt_image_array, lcmt_trajectory_block

class LCMStuff:
    def __init__(self):
        return

# Note: this method needs to be wrapped in a lock because the image and state can be
# modified at any time by the listener thread
def depthImgtoPoints(image, K, pq, c2w):
    points = np.array(image.shape, 3)
    for u in range(image.shape[0]):
        for v in range(image.shape[1]):
            point = np.linalg.inv(K) @ [u, v, image[u, v]]
            point = np.array([point[0], point[1], point[2], 1])
            xyz = pt.transform(c2w, point)
            points[u,v] = xyz
    return points


# Note: this method needs to be wrapped in a lock because the image and state can be
# Compute the edge magnitude along both x and y axes.
def computeGradientMap(image):
    gradients = filters.sobel(image, mode = "same")
    return gradients


# spiral search to find the nearest safe step location
def findNearestSafeLocation(s_nom, tree):
    return


# In a loop: 
#   Wait for footstep (trajectory) message
#   Build a point cloud from the last image message (or, if the image isn't updated, use the cached one)
#   Filter the depth image and associate each point with a sobel gradient in the image
#   Check if the planned footstep is too close to a high gradient or out of bounds region
#   If it is too close, then move it to the closest safe location using a spiral search
#   If it is fine, then query the point cloud to get the true z value of the step,
#   and output the planned (x,y,z) location in an lcmt_saved_traj.
def main():
    last_image = None
    last_state = None
    state_loc, image_lock = threading.Lock(), threading.Lock()
    traj_lock = threading.Lock()

    self.lc = lcm.LCM()
    self.stop_listener = threading.Event()
    self.sub_state = self.lc.subscribe(state_channel, self.state_handler)
    self.sub_images = self.lc.subscribe("DRAKE_RGBD_CAMERA_IMAGES", self.image_handler)
    
    # Handling function for depth image messages
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
        # convert from mm to m
        # 0 and 65.536 are reserved quantities for out of bounds.
        image = image.astype(np.float32)/1000
        self.image_queue.put(image)


    # Handler to get the nominal foot locations from the trajectory
    def trajectory_handler(channel, data):
        msg = lcmt_saved_traj.decode(data)
        if msg.num_trajectories > 0:
            sft = msg.trajectories[list(msg.trajectory_names).index("swing_foot_traj")]
            point = np.array(list(sft.datapoints))[:,0].reshape(3)
        # set a global point variable or something (with locking)
        return 


    # Handler to get the robot state to build a calibrated point cloud
    def state_handler(channel, data):
        msg = lcmt_robot_output.decode(data)
        pos = list(msg.position[0:7])
        # p,q 
        state = np.array(pos[4:7] + pos[0:4])
        state_queue.put(state)


    def lcm_listener(self):
        while True:
            lc.handle() 
            if stop_listener.is_set():
                break
            time.sleep(0.001)

    while True:
        try:
            # doo the loop in here.
        except KeyboardInterrupt:
    return


if __name__ == "__main__":
    main()
