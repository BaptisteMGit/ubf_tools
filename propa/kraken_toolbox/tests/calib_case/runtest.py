import os
from propa.kraken_toolbox.run_kraken import run_kraken_exec
from propa.kraken_toolbox.plot_utils import plotshd, plotmode
import matplotlib.pyplot as plt

working_dir = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\calib_case"
template_env = "calibK"
os.chdir(working_dir)

run_kraken_exec(template_env)
plotshd(os.path.join(working_dir, template_env + ".shd"))
plotmode(os.path.join(working_dir, template_env))

plt.show()
