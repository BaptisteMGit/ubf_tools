"""
Directivity patterns of different types of antennas 
"""

import numpy as np
import matplotlib.pyplot as plt

# Global variables
c0 = 1500  # speed of sound in water (m/s)
frequency = 5  # frequency (Hz)
n_sensors = 2  # number of sensors
d_inter_sensor = 20 * 1e3  # inter-sensor distance (m)


def linear_beampattern(
    n_sensors, d_inter_sensor, n_angles, angle_range, angle_unit, frequency, c0
):
    """Return the directivity pattern of a linear array of n_sensors sensors."""

    wl = c0 / frequency  # wavelength

    if angle_unit == "deg":
        angle_range = np.deg2rad(angle_range)
    elif angle_unit == "rad":
        pass
    else:
        raise ValueError('angle_unit must be "deg" or "rad"')

    # Assert n_angles is an integer
    if n_angles != int(n_angles):
        n_angles = int(n_angles)
    else:
        pass

    theta = np.linspace(angle_range[0], angle_range[1], n_angles)  # angle in rad
    psi_theta = np.pi * d_inter_sensor / wl * np.sin(theta)

    beampattern = (np.sin(n_sensors * psi_theta) / (n_sensors * np.sin(psi_theta))) ** 2

    return theta, beampattern

    # Directivity knots


def directivity_knots(
    theta,
    beampattern,
    n_sensors,
    d_inter_sensor,
    frequency,
    c0,
):
    """Derive directivity knots positions and values."""

    wl = c0 / frequency  # wavelength
    psi_theta = np.pi * d_inter_sensor / wl * np.sin(theta)
    beampattern = (np.sin(n_sensors * psi_theta) / (n_sensors * np.sin(psi_theta))) ** 2

    L = d_inter_sensor * n_sensors
    k_max, k_min = np.floor([L / wl, -L / wl])
    k_list = np.arange(k_min, k_max + 1, dtype=int)
    directivity_knots = [np.arcsin(k * wl / L) for k in k_list if k % n_sensors != 0]

    return directivity_knots


def plot_beampattern(theta, beampattern, title):
    """Plot directivity pattern."""

    # Derive knots positions
    knots, knots_values = directivity_knots(
        theta,
        beampattern,
        n_sensors,
        d_inter_sensor,
        frequency,
        c0,
    )

    # Log linear plot
    plt.figure()
    plt.plot(np.rad2deg(theta), 10 * np.log10(beampattern))
    plt.vlines(
        x=np.rad2deg(knots),
        ymin=min(10 * np.log10(beampattern)),
        ymax=0,
        color="r",
        linestyle="--",
        alpha=0.1,
    )
    plt.xlabel("Angle (deg)")
    plt.ylabel("Amplitude (dB)")
    plt.xlim([-90, 90])
    plt.title(title)
    # plt.show()

    # Polar plot
    plt.figure()
    plt.polar(theta, 10 * np.log10(beampattern))
    ax = plt.gca()
    ax.vlines(
        x=knots,
        ymin=min(10 * np.log10(beampattern)),
        ymax=0,
        color="r",
        linestyle="--",
        alpha=0.1,
    )

    ax.set_theta_offset(np.pi / 2)
    ax.set_thetamin(90)
    ax.set_thetamax(-90)

    plt.title(title)
    plt.ylabel("Amplitude (dB)")
    plt.show()

    return


if __name__ == "__main__":
    ns = 2
    f = 5
    c0 = 1500
    d = 20 * 1e3  # SWIR
    # d = 512
    wl = c0 / f
    # d = 5 * wl
    n_angles = 36000
    angle_range = [-180, 180]

    # -3 dB beamwidth (deg)
    two_theta_m3dB = 0.88 * wl / (ns * d)
    print(f"3 dB beamwidth: {np.round(np.rad2deg(two_theta_m3dB), 2)} deg")

    # Grid size at distance D from the array center
    D = 50 * 1e3
    dx = 2 * D * np.tan(two_theta_m3dB / 2)
    print(
        f"Grid size at distance D = {D*1e-3}km from the array center: {np.round(dx)} m"
    )

    # idx = np.argmin(np.abs(linear_bp - 0.5))
    # print(f"3 dB beamwidth: {np.rad2deg(theta[idx])} deg")

    theta, linear_bp = linear_beampattern(ns, d, n_angles, angle_range, "deg", f, c0)
    plot_beampattern(
        theta,
        linear_bp,
        f"Linear array beampattern \n ({f} Hz, {ns} sensors, d = {d} m)",
    )


# # Secondary lobes
# def secondary_lobes(
#     n_sensors, d_inter_sensor, n_angles, angle_range, angle_unit, frequency, c0
# ):
#     """Derive secondary lobes positions and values."""

#     wl = c0 / frequency  # wavelength
#     a = np.pi * d_inter_sensor / wl
#     psi_theta = a * np.sin(theta)

#     # Find secondary lobes defined by local maximums
#     beampattern_p = 1 / n_sensors**2 * 2 * a * np.cos(theta) * np.sin(
#         n_sensors * psi_theta
#     ) * 1 / np.sin(psi_theta) ** 2 * n_sensors * np.cos(n_sensors * psi_theta) - np.sin(
#         n_sensors * psi_theta
#     ) * 1 / np.tan(
#         psi_theta
#     )  # First derivative of beampattern
