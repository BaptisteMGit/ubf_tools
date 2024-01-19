import os
import sys


# sys.path.append(r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\pyAT")

import matplotlib.pyplot as plt
from propa.kraken_toolbox import utils, read_modes, plot_utils


working_dir = (
    r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\testPyAT"
)

os.chdir(working_dir)

filename = "MunkK"

utils.runkraken(filename)

filename = "MunkK.shd"
# modes = read_modes.readmodes(filename, 0, [1, 10, 30, 60])
plot_utils.plotshd(filename, units="km", freq=700)
# plot_utils.plotmode(filename, 0, [1, 10, 30, 60])

plt.show()
