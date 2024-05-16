import os
import matplotlib.pyplot as plt
from propa.kraken_toolbox.utils import runkraken
from propa.kraken_toolbox.plot_utils import plotshd


working_dir = (
    r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\step"
)
env_filename = "stepK_rd"
os.chdir(working_dir)
# runkraken(env_filename)

plotshd(env_filename + ".shd", tl_min=60, tl_max=110, title="Step K")
# plt.clim(60, 110)
plt.show()
