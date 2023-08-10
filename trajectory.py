import json
import h5py
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QApplication, QWidget, QSlider, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QLineEdit, QMessageBox, QCheckBox, QFrame, QTabWidget, QMainWindow, QTableWidget, QHeaderView, QTableWidgetItem, QFileDialog
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

from utils import process_frame
import utils 


class TrajectoryTab(QWidget):
    def __init__(self, main_window, parent=None):
        super(TrajectoryTab, self).__init__(parent)
        
        main_layout = QHBoxLayout(self)

        self.load_pushbutton = QPushButton('Load geezer centroids')
        self.load_pushbutton.clicked.connect(self.load)

        self.eye_box = QVBoxLayout()
        self.cam_box = QVBoxLayout()
        self.led_box = QVBoxLayout()
        self.fids_box = QHBoxLayout()

        boxes = [self.eye_box, self.cam_box, self.led_box]

        for box in boxes:
            self.fids_box.addLayout(box)
        
        main_layout.addLayout(self.fids_box)
        main_layout.addWidget(self.load_pushbutton)
        self.setLayout(main_layout)


    def load(self):
        # Popup window to select file
        self.filename = QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        self.filename = str(self.filename)
        with h5.File(self.filename, 'r') as f:
            self.keys = list(f.keys())

            frame_idxs = f['frame_idxs'][:]  
            sidx = np.argsort(frame_idxs)
            frame_idxs = frame_idxs[sidx]

            pup_co = f['pup_co'][:][sidx]
            for i,(k,v) in enumerate(f['fids_co'].items()):
                print(k)
                print(v)



