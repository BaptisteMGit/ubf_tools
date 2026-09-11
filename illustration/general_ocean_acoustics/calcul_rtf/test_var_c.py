import numpy as np
import matplotlib.pyplot as plt

from source.normal_modes import image_sources_arrivals

c0 = 1500
z_s = 5
z_r = 99
D = 100
d12 = 1000
r1 = 5 * 1e3
r2 = r1 + d12

# Receiver 1
tm1 = image_sources_arrivals(z_src=z_s, z_rcv=z_r, r=r1, depth=D, n=1, c0=c0)
Rm1 = tm1[0] * c0
# Receiver 2
tm2 = image_sources_arrivals(z_src=z_s, z_rcv=z_r, r=r2, depth=D, n=1, c0=c0)
Rm2 = tm2[0] * c0


A1 = 1 / Rm1[0]  # R01(r)
A2 = 1 / Rm2[0]  # R01(r+d12)
B1 = 1 / Rm1[2]  # R03(r)
B2 = 1 / Rm2[2]  # R03(r+d12)

f = np.linspace(5, 150, 10000)
k = c0 / f
H1_mod = np.sqrt(A1**2 + B1**2 - 2 * A1 * B1 * np.cos(k * (1 / A1 - 1 / B1)))
H2_mod = np.sqrt(A2**2 + B2**2 - 2 * A2 * B2 * np.cos(k * (1 / A2 - 1 / B2)))
rtf = H2_mod / H1_mod
print(A1, B1, A2, B2)
# Ajout d'une perturbation
eps_c = 20
c_tilde = c0 + eps_c
k_tilde = c_tilde / f

# Receiver 1
tm1 = image_sources_arrivals(z_src=z_s, z_rcv=z_r, r=r1, depth=D, n=1, c0=c_tilde)
Rm1 = tm1[0] * c_tilde
# Receiver 2
tm2 = image_sources_arrivals(z_src=z_s, z_rcv=z_r, r=r2, depth=D, n=1, c0=c_tilde)
Rm2 = tm2[0] * c_tilde

A1 = 1 / Rm1[0]  # R01(r)
A2 = 1 / Rm2[0]  # R01(r+d12)
B1 = 1 / Rm1[2]  # R03(r)
B2 = 1 / Rm2[2]  # R03(r+d12)

H1_mod_tilde = np.sqrt(
    A1**2 + B1**2 - 2 * A1 * B1 * np.cos(k_tilde * (1 / A1 - 1 / B1))
)
H2_mod_tilde = np.sqrt(
    A2**2 + B2**2 - 2 * A2 * B2 * np.cos(k_tilde * (1 / A2 - 1 / B2))
)
rtf_tilde = H2_mod_tilde / H1_mod_tilde


fig, axs = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
axs[0].plot(f, H1_mod)
axs[0].plot(f, H1_mod_tilde, label=r"$\delta _c$")
axs[1].plot(f, H2_mod)
axs[1].plot(f, H2_mod_tilde, label=r"$\delta _c$")
axs[2].plot(f, rtf)
axs[2].plot(f, rtf_tilde, label=r"$\delta _c$")

axs[2].set_yscale("log")
fig.supxlabel("Fréquence [Hz]")

plt.show()
