# https://lcm-proj.github.io/tut_python.html
import lcm
from dairlib import lcmt_robot_output
import subprocess as sp

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


if __name__ == "__main__":
    main()
