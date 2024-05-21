import os
import matplotlib.pyplot as plt
from propa.kraken_toolbox.run_kraken import run_kraken_exec
from propa.kraken_toolbox.plot_utils import plotshd


working_dir = (
    r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\step"
)
env_filename = "stepK_rd"
os.chdir(working_dir)
run_kraken_exec(env_filename)

plotshd(env_filename + ".shd", tl_min=60, tl_max=110)
img_path = os.path.join(working_dir, f"tl_{env_filename}.png")
plt.savefig(img_path)
