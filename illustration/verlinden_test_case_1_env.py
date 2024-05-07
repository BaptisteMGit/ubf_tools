import numpy as np
import matplotlib.pyplot as plt
from propa.kraken_toolbox.kraken_env import KrakenMedium

sig_type = "pulse_train"

z_ssp = [0, 150]
cp_ssp = [1500, 1500]

medium = KrakenMedium(ssp_interpolation_method="C_linear", z_ssp=z_ssp, c_p=cp_ssp)

medium.plot_medium()
plt.show()

bathy = np.array(
    [
        3500,
        3560,
        3550,
        3500,
        3560,
        3555,
        3500,
        3530,
        3538,
        3509,
        3534,
        3550,
        3549,
        3560,
        3555,
        3600,
        3750,
        3950,
        3960,
        3975,
        3990,
        4500,
        4560,
        4550,
        4500,
        4560,
        4555,
        4500,
        4530,
        4538,
        5010,
        5020,
        5045,
        5036,
        5010,
        5020,
        5045,
        5036,
    ]
)

r = np.linspace(0, 50, bathy.size)

label_fontsize = 20
plt.figure(figsize=(16, 8))
plt.plot(r, bathy, color="k")
plt.xlabel("Range [km]", fontsize=label_fontsize)
plt.ylabel("Depth [m]", fontsize=label_fontsize)
plt.ylim([0, 5500])
plt.xlim([0, 50])
ax = plt.gca()
ax.fill_between(r, bathy, 5500, color="peachpuff")
ax.fill_between(r, 0, bathy, color="lightblue")
ax.invert_yaxis()

plt.yticks(fontsize=label_fontsize)
plt.xticks(fontsize=label_fontsize)


# Munkk ssp profile
c = np.array(
    [
        1548.52,
        1530.29,
        1526.69,
        1517.78,
        1509.49,
        1504.30,
        1501.38,
        1500.14,
        1500.12,
        1501.02,
        1502.57,
        1504.62,
        1507.02,
        1509.69,
        1512.55,
        1515.56,
        1518.67,
        1521.85,
        1525.10,
        1528.38,
        1531.70,
        1535.04,
        1538.39,
        1541.76,
        1545.14,
        1548.52,
        1551.91,
    ]
)

z = np.array(
    [
        0.0,
        200.0,
        250.0,
        400.0,
        600.0,
        800.0,
        1000.0,
        1200.0,
        1400.0,
        1600.0,
        1800.0,
        2000.0,
        2200.0,
        2400.0,
        2600.0,
        2800.0,
        3000.0,
        3200.0,
        3400.0,
        3600.0,
        3800.0,
        4000.0,
        4200.0,
        4400.0,
        4600.0,
        4800.0,
        5000.0,
    ]
)

plt.figure(figsize=(8, 10))
plt.plot(c, z, color="k", linewidth=3)
plt.xlabel(r"Sound speed [$m.s^{-1}$]", fontsize=label_fontsize)
plt.ylabel("Depth [m]", fontsize=label_fontsize)
plt.ylim([min(z), max(z)])
plt.xlim([min(c) - 5, max(c) + 5])
ax = plt.gca()
# # ax.fill_between(r, bathy, 5500, color="peachpuff")
# ax.fill_between(c, 0, z, color="lightblue")
# ax.fill_between(c, 5000, z, color="lightblue")
ax.fill(
    [min(c) - 5, min(c) - 5, max(c) + 5, max(c) + 5],
    [min(z), max(z), max(z), min(z)],
    color="lightblue",
)
ax.invert_yaxis()

plt.yticks(fontsize=label_fontsize)
plt.xticks(fontsize=label_fontsize)
plt.tight_layout()
plt.show()
