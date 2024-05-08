import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.figure import Figure
import IPython
import h5py as h5
# import utils 


import IPython

from PyQt5.QtWidgets import QApplication, QWidget, QSlider, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QLineEdit, QMessageBox, QCheckBox, QFrame, QTabWidget, QMainWindow, QTableWidget, QHeaderView, QTableWidgetItem
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, 
    QDialog, QComboBox, QLineEdit, QHBoxLayout, QLabel
)
import sys
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas
)
from matplotlib.figure import Figure

from PyQt5.QtGui import QIcon
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import os
import threading
import json
import h5py as h5
import multiprocessing as mp
import tqdm
import pickle
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure 
# Description: Enable capability of selecting starting fiducials on one machine, saving a file, then running the image_proc algorithm on that mp4 and those fiducials (via the file) on another machine. This will enable users to perform the necessary GUI-dependent functions on laptop, then run the heavy-compute processes on a server (i.e. foghorn)
import cv2
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt

# import utils 

# from geometry import GeometryTab

# rcParams: set default so that axis isn't shown and no x or yticks
plt.rcParams['axes.axisbelow'] = False
plt.rcParams['xtick.bottom'] = False
plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['ytick.left'] = False
plt.rcParams['ytick.labelleft'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.bottom'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

f = h5.File('/home/yfaragal/07062023/July062023jf/4-10-2024-goldroger-89k-91k-geezer.h5','r')
# f = h5.File('/data/cortex/raw/GolDRoger/goldroger_geezer.h5','r')
print(f['fiducial_coordinates'].keys())
# f0 = f['fiducial_coordinates']['cam']
# f1 = f['fiducial_coordinates']['ne']
f2 = f['fiducial_coordinates']['se']
timess = np.arange(f2.shape[0])
# Description: Enable capability of selecting starting fiducials on one machine, saving a file, then running the image_proc algorithm on that mp4 and those fiducials (via the file) on another machine. This will enable users to perform the necessary GUI-dependent functions on laptop, then run the heavy-compute processes on a server (i.e. foghorn)
f3 = f['fiducial_coordinates']['sw']
# f4 = f['fiducial_coordinates']['fid_4']
pup1 = f['pup_co'][:,0]
pup2 = f['pup_co'][:,1]
# ax1 = plt.subplot(411)
# plt.plot(timess,f0, label='cam')
# plt.tick_params('x', labelsize=6)
# plt.legend()
# ax2 = plt.subplot(412, sharex=ax1)
# plt.plot(timess,f1, label='ne')
# plt.legend()
ax3 = plt.subplot(411)
plt.plot(timess,f2, label='sw')
plt.legend()
ax4 = plt.subplot(412, sharex=ax3)
plt.plot(timess,f3, label='se')
plt.legend()
# ax5 = plt.subplot(615, sharex=ax1)
# plt.plot(timess,f4, label='sw')
# plt.legend()
ax5 = plt.subplot(413, sharex=ax3)
plt.plot(timess,pup1, label='pup_co_x')
plt.legend()
ax6 = plt.subplot(414, sharex=ax3)
plt.plot(timess,pup2, label='pup_co_y')
plt.tick_params('x', labelsize=6)
plt.legend()
plt.show()
# old way for 07062023 fids_co/fid_'n'