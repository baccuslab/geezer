%load_ext autoreload
%autoreload 2

# %%
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('/home/melandrocyte/Code/geezer/src/')
import utils

FILEPATH = '/home/melandrocyte/imu_geezer_output_round_2.h5'

# %%
file = h5.File(FILEPATH, 'r')

se_coordinates = file['fiducial_coordinates']['se']
sw_coordinates = file['fiducial_coordinates']['sw']

plt.plot(se_coordinates[:,0] - se_coordinates[0,0])
plt.plot(sw_coordinates[:,0] - sw_coordinates[0,0])
plt.show()

# %%
