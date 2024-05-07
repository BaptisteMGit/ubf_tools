import os

dir = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\test_bug_az"
os.chdir(dir)
fname = r"testcase2_1"
os.system(f"kraken {fname}")
os.system(f"field {fname}")
#
