import matplotlib.pyplot as plt
from propa.kraken_toolbox.kraken_env import KrakenMedium

sig_type = "pulse_train"

z_ssp = [0, 150]
cp_ssp = [1500, 1500]

medium = KrakenMedium(ssp_interpolation_method="C_linear", z_ssp=z_ssp, c_p=cp_ssp)

medium.plot_medium()
plt.show()
