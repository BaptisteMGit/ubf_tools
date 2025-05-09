#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   antenna.py
@Time    :   2025/05/06 11:40:26
@Author  :   Menetrier Baptiste
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   Class to handle antenna properties
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import numpy as np
import matplotlib.pyplot as plt


class Antenna:
    """
    Class to handle antenna properties
    """

    def __init__(self, name: str, n_elements: int):
        """
        Constructor
        :param name: Name of the antenna
        :param n_elements: Number of elements in the antenna
        """
        self.name = name
        self.n_elements = n_elements
        self.x = None
        self.y = None
        self._antenna_radius = None
        self._rcv_idx = None

    @property
    def antenna_radius(self):
        self._antenna_radius = np.max(np.sqrt(self.x**2 + self.y**2))
        return self._antenna_radius

    @antenna_radius.setter
    def antenna_radius(self, value):
        """
        Set the antenna radius
        :param value: Antenna radius
        """
        if value <= 0:
            raise ValueError("Antenna radius must be positive")
        self._antenna_radius = value

    @property
    def rcv_idx(self):
        """
        Get the receiver index
        :return: Receiver index
        """
        self._rcv_idx = np.arange(self.n_elements)
        return self._rcv_idx

    def plot_antenna(self):
        """
        Plot the antenna
        """

        plt.figure()
        # Add arrows to indicate the reference frame
        alpha = 0.5
        plt.quiver(
            0,
            0,
            self.antenna_radius * alpha,
            0,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="r",
            label="X",
        )
        plt.quiver(
            0,
            0,
            0,
            self.antenna_radius * alpha,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="b",
            label="Y",
        )
        # Add anotation to the arrows
        plt.text(
            self.antenna_radius * alpha * 1.1,
            0,
            "X",
            fontsize=12,
            color="r",
            ha="center",
            va="top",
        )
        plt.text(
            0,
            self.antenna_radius * alpha * 1.1,
            "Y",
            fontsize=12,
            color="b",
            ha="right",
            va="center",
        )

        # Add origin
        # plt.scatter(0, 0, color="k", marker="o")

        # Add the antenna elements
        plt.scatter(self.x, self.y, color="k", marker="o")
        # Add receivers ids
        for i in range(self.n_elements):
            plt.text(
                self.x[i],
                self.y[i],
                str(i),
                fontsize=12,
                color="k",
                ha="left",
                va="bottom",
            )

        r = 1.1 * self.antenna_radius
        plt.xlim([-r, r])
        plt.ylim([-r, r])
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title(f"Antenna {self.name}")
        plt.grid()
        # plt.show()


class SparseAntenna(Antenna):
    """
    Sparse antenna class
    """

    def __init__(
        self, name: str, n_elements: int, random_radius: float, rng_seed: int = 42
    ):
        """
        Constructor
        :param name: Name of the antenna
        :param n_elements: Number of elements in the antenna
        :param random_radius: Radius from origin to randomize the position of the elements
        """
        super().__init__(name, n_elements)

        self.rng_seed = rng_seed
        rng = np.random.default_rng(seed=rng_seed)
        self.x = rng.uniform(-random_radius, random_radius, n_elements)
        self.y = rng.uniform(-random_radius, random_radius, n_elements)

        self.order_receivers()

    def order_receivers(self):
        """Order receivers according to their coordinates"""

        theta = np.arctan2(self.y, self.x)
        idx = np.argsort(theta)
        self.x = self.x[idx]
        self.y = self.y[idx]


class LinearAntenna(Antenna):
    """
    Linear antenna class
    """

    def __init__(self, name: str, n_elements: int, spacing: float):
        """
        Constructor
        :param name: Name of the antenna
        :param n_elements: Number of elements in the antenna
        :param spacing: Spacing between the elements
        """
        super().__init__(name, n_elements)

        self.x = np.arange(n_elements) * spacing
        self.y = np.zeros(n_elements)


if __name__ == "__main__":
    nrcv = 6
    r = 5 * 1e3  # 5 km
    antenna = SparseAntenna("SparseAntenna", nrcv, r)
    print(antenna.x)
    print(antenna.y)
    antenna.plot_antenna()

    delta = 500  # 500 m
    l_antenna = LinearAntenna("LinearAntenna", nrcv, delta)
    print(l_antenna.x)
    print(l_antenna.y)
    l_antenna.plot_antenna()

    plt.show()
