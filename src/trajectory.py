import json
import pickle
import h5py
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QApplication, QWidget, QSlider, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QLineEdit, QMessageBox, QCheckBox, QFrame, QTabWidget, QMainWindow, QTableWidget, QHeaderView, QTableWidgetItem, QFileDialog, QListWidget
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

from utils import process_ellipse
import utils 


class TrajectoryTab(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__()
        self.combo_boxes = {}
        self.combo_buttons = {}
        
        self.geometry_filename = None
        self.main_window = main_window
        self.coord_path = ""
        self.geo_path = ""
        
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Push Button to load coordinates
        load_button = QPushButton('Load Coords')
        load_button.clicked.connect(self.load_coords)
        layout.addWidget(load_button)

        # Labels for coord_path and geo_path
        self.coord_label = QLabel(f"Coord Path: {self.coord_path}")
        self.geo_label = QLabel(f"Geo Path: {self.geo_path}")
        layout.addWidget(self.coord_label)
        layout.addWidget(self.geo_label)

        # Columns for image processing and geometry entitites
        columns = ['Image processing', 'Geometry']
        self.selection_layout = QHBoxLayout()

        # Create list for image processing entities
        self.image_processing_list = QListWidget()

        # Create list for geometry entities
        self.geometry_list = QListWidget()

        # Add lists to selection layout
        self.selection_layout.addWidget(self.image_processing_list)
        self.selection_layout.addWidget(self.geometry_list)

    def load_coords(self):
        # Popup window to select file
        self.coord_filename = QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        self.coord_label.setText(self.coord_filename)
        
        self.fiducial_coords = {}

        with h5.File(self.coord_filename, 'r') as f:
            true_frame_idxs = f['frame_idxs'][:]
            sorted_frame_idxs = np.argsort(true_frame_idxs)


            self.fiducial_keys = list(f['fiducial_coords'].keys())

            for key in self.fiducial_keys:
                self.fiducial_coords[key] = np.array(f['fiducial_coords'][key][:])[sorted_frame_idxs]

            self.fiducial_coords['pupil'] = np.array(f['pup_coords'][:])[sorted_frame_idxs]




