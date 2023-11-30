""" Class for simulation using the Verlinden model."""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import sys
from propa.PyAT.kraken import runkraken


class VerlindenSimu:
    def __init__(
        self,
        c0=1500,
        fmin=1,
        fmax=50,
        df=1,
        n_sensors=2,
        d_inter_sensor=20 * 1e3,
        localisation_distance=50 * 1e3,
    ):
        """Initialise the simulation."""
        self.c0 = c0  # speed of sound in water (m/s)
        self.fmin = fmin  # min frequency (Hz)
        self.fmax = fmax  # max frequency (Hz)
        self.df = 1  # frequency step (Hz)
        self.frequencies = np.arange(self.fmin, self.fmax, self.df)  # frequencies (Hz)
        self.n_sensors = n_sensors  # number of sensors
        self.d_inter_sensor = d_inter_sensor  # inter-sensor distance (m)
        self.wl = self.c0 / self.frequencies  # wavelength
        self.array_length = (
            self.n_sensors * self.d_inter_sensor
        )  # total array length (m)
        self.localisation_distance = localisation_distance  # localisation distance (m)

        # Approximate size of the SWIR array = 100 km * 100 km
        self.Lx = 100 * 1e3
        self.Ly = 100 * 1e3
        self.Hmax = 5 * 1e3  # max depth (m)

        self.derive_grid_size()
        self.init_grid()
        self.init_environment()

    def derive_grid_size(self):
        """Derive the minimum grid size to ensure signal coherence within a cell from the array geometry and the localisation distance"""

        two_theta_m3dB = 0.88 * self.wl / self.array_length  # -3 dB beamwidth (deg)
        self.dx = 2 * self.localisation_distance * np.tan(two_theta_m3dB / 2)
        self.dx = round(min(self.dx), 0)
        self.dy = self.dx
        self.dz = 5  # Change depending on the propagation code requirements and the celerity profile resolution

    def init_grid(self):
        self.grid_x = np.arange(-self.Lx / 2, self.Lx / 2, self.dx)
        self.grid_y = np.arange(-self.Ly / 2, self.Ly / 2, self.dy)
        self.grid_z = np.arange(0, self.Hmax, self.dz)

    def init_environment(self):
        self.environement_dataset = xr.Dataset(
            data_vars=dict(),
            coords=dict(
                x=("x", self.grid_x),
                y=("y", self.grid_y),
                z=("z", self.grid_z),
            ),
            attrs=dict(
                description="Environement dataset containing the grid and other ocean properties"
            ),
        )


class Environement:
    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy


if __name__ == "__main__":
    c0 = 1500
    fmin, fmax, df = 5, 50, 5
    verlinden = VerlindenSimu(c0=c0, fmin=fmin, fmax=fmax, df=df)
    print(len(verlinden.grid_x))
    print(verlinden.grid_y)
