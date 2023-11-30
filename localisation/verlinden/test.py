import numpy as np
import matplotlib.pyplot as plt
from AcousticComponent import AcousticSource, ReceiverArray
from signals import sine_wave, ship_noise

# Source

y, t = ship_noise()

# y, t = sine_wave(f0=10, fs=150, T=1, A=1, phi=0)

Source = AcousticSource(y, t)
Source.display_source()

rcv_names = ["rcv1", "rcv2", "rcv3", "rcv4"]
H = 50
rcv_pos = [[0, 0, H], [0, 100, H], [100, 0, H], [100, 100, H]]

rcv_array = ReceiverArray(rcv_names, rcv_pos)
rcv_array.plot_array()


plt.show()
