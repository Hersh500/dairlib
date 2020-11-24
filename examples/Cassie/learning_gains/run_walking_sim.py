import subprocess
import time

import numpy as np

# You can run multiple processes!
# cmd1 = ['bazel-bin/examples/Cassie/run_sim_and_walking',
#                   '--end_time=5'
#                   ]
# process1 = subprocess.Popen(cmd1)
# time.sleep(1)

cmd = ['bazel-bin/examples/Cassie/run_sim_and_walking',
                  '--end_time=5'
                  ]
process = subprocess.Popen(cmd)

while process.poll() is None:
  # subprocess is alive
  time.sleep(1)

print("end of python script")
