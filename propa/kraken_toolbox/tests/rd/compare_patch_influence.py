import os
import pandas as pd
import matplotlib.pyplot as plt

root = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\rd"
name = "tl_along_range_semiflat_ref"
path = os.path.join(root, name + ".csv")
pd1 = pd.read_csv(path)

name = "tl_along_range_semiflat_5000"
path = os.path.join(root, name + ".csv")
pd2 = pd.read_csv(path)


plt.figure()
plt.plot(pd1["range"], pd1["tl"], label="Ref")
plt.plot(pd2["range"], pd2["tl"], label="Patch - 3500")

plt.xlabel("Range [km]")
plt.ylabel("TL [dB]")
plt.gca().invert_yaxis()
plt.legend()
plt.show()
