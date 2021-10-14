# https://lcm-proj.github.io/tut_python.html
import lcm
from dairlib import lcmt_robot_output

def handler(channel, data):
    msg = lcmt_robot_output.decode(data)
    print("Received message!")
    print("timestamp = %s" % str(msg.utime))

lc = lcm.LCM()
sub = lc.subscribe("CASSIE_STATE_SIMULATION", handler)

try:
    while True:
        lc.handle()
except KeyboardInterrupt:
    pass
