# Takes a nominal footstep through an lcmt_saved_traj
# and an image message, and then plans a footstep based on that
import lcm
import numpy as np
import argparse
from skimage import filters
from scipy.spatial import KDTree
import threading
from pytransform3d import transformations as pt
from pytransform3d import rotations as pr
import queue
from dairlib import lcmt_saved_traj, lcmt_robot_out, lcmt_image_array, lcmt_trajectory_block


# Communicates with MPC over LCM
class MPCInterface:
    def __init__(self, state_channel, image_channel, traj_channel):
        self.lc = lcm.LCM()
        self.stop_listener = threading.Event()

        self.sub_state = self.lc.subscribe(state_channel, self.state_handler)
        self.sub_images = self.lc.subscribe(image_channel, self.image_handler)
        self.sub_traj = self.lc.subscribe(traj_channel, self.traj_handler)
        self.image_queue = queue.LifoQueue()
        self.state_queue = queue.LifoQueue()
        self.traj_queue = queue.LifoQueue()

        # spawn a thread that listens over LCM
        self.listener_thread = threading.Thread(target=self.lcm_listener)
        self.listener_thread.start()


    def lcm_listener(self):
        while True:
            lc.handle() 
            if stop_listener.is_set():
                break
            time.sleep(0.01)


    # Handling function for depth image messages
    def image_handler(self, channel, data):
        msg = lcmt_image_array.decode(data)
        image_msg = msg.images[0]
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
    def trajectory_handler(self, channel, data):
        msg = lcmt_saved_traj.decode(data)
        if msg.num_trajectories > 0:
            sft = msg.trajectories[list(msg.trajectory_names).index("swing_foot_traj")]
            point = np.array(list(sft.datapoints))[:,0].reshape(3)
            self.traj_queue.put(point)


    # Handler to get the robot state to build a calibrated point cloud
    def state_handler(self, channel, data):
        msg = lcmt_robot_output.decode(data)
        pos = list(msg.position[0:7])
        # p,q
        state = np.array(pos[4:7] + pos[0:4])
        self.state_queue.put(state)


def depthImgtoPoints(image, K, c2w):
    points = np.zeros((image.shape[0], image.shape[1], 3)
    for u in range(image.shape[0]):
        for v in range(image.shape[1]):
            z_val = image[u, v]
            point = np.linalg.inv(K) @ np.array([v, u, 1]) * z_val
            point = np.array([point[0], point[1], point[2], 1])
            xyz = pt.transform(c2w, point)[0:3]
            points[u,v] = xyz
    return points


def pointToUV(p_w, K, w2c):
    p_c = pt.transform(w2c, p_w)
    p_im = K@p_c[0:3]/p_c[2]
    return p_im

def computeEdgeMap(image):
    gradients = filters.sobel(image, mode = "same")
    return gradients


def processImage(image, K, c2w):
    points_image = depthImgtoPoints(image, K, body_pq, c2b)
    edges = np.expand_dims(computeEdgeMap(image), axis = 2)
    all_feats = np.concatenate([points_image, edges], axis = 2)
    return all_feats


# TODO(hersh500): figure out reasonable gradient values.
def findNearestSafeLocation(uv, all_feats, z_bounds, gradient_max = 0.1, safety_radius = 5):
    def is_good(u, v):
        nearby_gradients = all_feats[u-safety_radius:u+safety_radius, v-safety_radius:v+safety_radius, 3]
        nearby_heights = all_feats[u-safety_radius:u+safety_radius, v-safety_radius:v+safety_radius, 2]
        # this means the nominal is okay.
        if (np.all(nearby_gradients <= gradient_max) and
                       np.all(nearby_heights <= z_bounds[1]) and
                       np.all(nearby_heights >= z_bounds[0])):
            return True
        return False
        
    sign = 1
    u = uv[0]
    v = uv[1]
    for delta in range(all_feats.shape[0]):
        for k in range(delta):
            u += delta * sign
            if is_good(u, v):
                return all_feats[u, v, :3]     
        for k in range(delta):
            v += -1 * delta * sign
            if is_good(u, v):
                return all_feats[u, v, :3]     
        sign *= -1

    # If we literally cannot find a safe location, just return the nominal 
    return all_feats[uv[0], uv[1], :3]
        
    
# In a loop: 
#   Wait for footstep (trajectory) message
#   Build a point cloud from the last image message (or, if the image isn't updated, use the cached one)
#   Filter the depth image and associate each point with a sobel gradient in the image
#   Check if the planned footstep is too close to a high gradient or out of bounds region
#   If it is too close, then move it to the closest safe location using a spiral search
#   If it is fine, then query the point cloud to get the true z value of the step,
#   and output the planned (x,y,z) location in an lcmt_saved_traj.
def main():
    parser = argparse.ArgumentParser(description='Receive step locations and check them.')
    parser.add_argument('low_lim', type=float, help='the low limit for step z deviation from 0', default = 0.1)
    parser.add_argument('high_lim', type=float, help='the high limit for step z deviation from 0', default = 0.2)
    parser.add_argument('edge_mag', type=float, help='the limit for safe edge magnitude', default = 0.1)
    args = parser.parse_args()
    z_low = parser.low_lim
    z_high = parser.low_lim
    edge_mag = parser.edge_mag

    c_x, c_y = 64, 64
    f_x, f_y = 100,100
    K = np.array([[f_x, 0, c_x],
                  [0, f_y, c_y],
                  [0, 0, 1]])
    camera_rot = pr.active_matrix_from_extrinsic_roll_pitch_yaw([-2.6, 0.0, -1.57])
    c2b = pt.transform_from(camera_rot, [0.05, 0, -0.15])
    leg_length = 0.8
    image = None

    while True:
        try:
            # wait till we get a nominal step from MPC and a state
            nom_step = interface.traj_queue.get(block = True)
            state = interface.state_queue.get(block = True)
            c2w = pt.concat(c2b, pt.transform_from_pq(state))
            body_z = state[2]
            if image is None:
                image = interface.image_queue.get(block = True)
                points = processImage(image, K, c2w)
            elif len(interface.image_queue) > 0:
                image = interface.image_queue.get()
                points = processImage(image, K, c2w)

            # If it's out of the image frame, we can't do anything.
            im_coords = pointToUV(nom_step, K, pt.invert_transform(c2w))
            if im_coords[0] < 0 or im_coords[1] < 0 or im_coords[1] > 128 or im_coords[0] > 128:
                best_step_loc = nom_step
            else:
                diffs = np.linalg.norm(points[:,:,0:2] - nom_step[0:2], axis = 2)
                best_uv = numpy.unravel_index(diffs.argmin(), diffs.shape)
                best_step_loc = findNearestSafeLocation(best_uv, all_feats, z_bounds = (body_z - leg_length - z_low, body_z - leg_length + z_high), edge_mag)

            # build output message
            msg = lcmt_saved_traj() 
            msg.num_trajectories = 1
            block = lcmt_trajectory_block()
            block.trajectory_name = "footstep_adj"
            block.num_points = 1
            block.num_datatypes = 3
            block.datapoints = list(best_step_loc.T)
            msg.trajectories = [block]
            msg.trajectory_names = ["footstep_adj"]
            interface.lc.publish("FOOTSTEP_PLANNER_OUT", msg.encode())
        except KeyboardInterrupt:
            interface.stop_listener.set()

if __name__ == "__main__":
    main()
