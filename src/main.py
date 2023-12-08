from PyQt5.QtWidgets import QApplication, QWidget, QSlider, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QLineEdit, QMessageBox, QCheckBox, QFrame, QTabWidget, QMainWindow, QTableWidget, QHeaderView, QTableWidgetItem, QVBoxLayout
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
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process
import multiprocessing

# from utils import process_frame
# import utils 

from geometry import GeometryTab
from trajectory import TrajectoryTab
from curate import CurateTab
from image_proc import ImageProcTab


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

DEFAULT_PARAMS = {'pupil': {'exp': 3, 'small': 20, 'large': 50, 'thresh': 200},
                  'fids': {'exp': 1, 'small': 3, 'large': 11, 'thresh': 65}}

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle('Geezer')
        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)

        self.geometry_tab = GeometryTab(self)
        self.trajectory_tab = TrajectoryTab(self)
        self.curate_tab = CurateTab(self)
        self.imageproc_tab = ImageProcTab(self)

        self.tabs.addTab(self.imageproc_tab, "Centroids")
        self.tabs.addTab(self.geometry_tab, "Geometry")
        self.tabs.addTab(self.trajectory_tab, "Trajectory")
        self.tabs.addTab(self.curate_tab, "Curate")

        self.tabs.currentChanged.connect(self.tab_changed)

    def tab_changed(self):
        # if the tab is trajectory
        if self.tabs.currentIndex() == 2:
            self.tabs.currentWidget().update()

def main():
    app = QApplication([])
    main_win = MainWindow()
    main_win.show()
    return app.exec_()

if __name__ == "__main__":
    main()

