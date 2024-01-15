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
        
        # Mapping key is image_proc and value is geometry
        self.mapping = {}
        self.mapping['led'] = {}
        self.mapping['camera'] = {}
        self.mapping['pupil'] = None

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
        
        # Lock buttons HBox
        lock_buttons_layout = QVBoxLayout()
        lock_buttons_layout.setAlignment(Qt.AlignLeft)
        lock_buttons_layout.setSpacing(10)
        
        led_lock_push_button = QPushButton('Lock LED')
        led_lock_push_button.clicked.connect(lambda: self.update_mapping('led'))

        camera_lock_push_button = QPushButton('Lock Camera')
        camera_lock_push_button.clicked.connect(lambda: self.update_mapping('camera'))

        pupil_lock_push_button = QPushButton('Lock Pupil')
        pupil_lock_push_button.clicked.connect(lambda: self.update_mapping('pupil'))

        lock_buttons_layout.addWidget(led_lock_push_button)
        lock_buttons_layout.addWidget(camera_lock_push_button)
        lock_buttons_layout.addWidget(pupil_lock_push_button)

        lock_buttons_layout.addStretch(1)

        # Execute button
        execute_button = QPushButton('Execute')
        execute_button.clicked.connect(self.execute)
        lock_buttons_layout.addWidget(execute_button)

        self.selection_layout.addLayout(lock_buttons_layout)

        # Add coordinate canvas
        
        self.quality_control_row = QHBoxLayout()
        self.quality_control_plot_layout = QVBoxLayout()
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.ax = self.figure.add_subplot(111)
        self.ax.invert_yaxis()

        self.quality_control_plot_layout.addWidget(self.toolbar)
        self.quality_control_plot_layout.addWidget(self.canvas)

        self.quality_control_row.addLayout(self.quality_control_plot_layout)
        self.quality_control_row.addStretch(1)






        # Add things to layout
        layout.addLayout(self.selection_layout)
        layout.addLayout(self.quality_control_row)
        layout.addStretch(1)
        self.setLayout(layout)


    def load_coords(self):
        if self.geometry_filename is None:
            QMessageBox.warning(self, 'Warning', 'Please select geometry file first.')
            return
        # Popup window to select file
        self.coord_filename = QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        self.coord_label.setText(self.coord_filename)
        
        self.fiducial_coords = {}
        with h5.File(self.coord_filename, 'r') as f:
            true_frame_idxs = f['frame_idxs'][:]
            # sorted_frame_idxs = np.argsort(true_frame_idxs)

            self.fiducial_keys = list(f['fiducial_coordinates'].keys())
        
        # Populate the lists 
        self.image_processing_list.clear()
        self.geometry_list.clear()

        for key in self.fiducial_keys:
            self.image_processing_list.addItem(key)
        
        self.image_processing_list.addItem('pupil')

        self.geometry_data = utils.load_geometry_json(self.geometry_filename)
        for key in list(self.geometry_data.keys()):
            self.geometry_list.addItem(key)

    def update_mapping(self, key):
        # Get the selected items from the lists
        selected_image_proc_item = self.image_processing_list.selectedItems()[0].text()
        selected_geometry_item = self.geometry_list.selectedItems()[0].text()

        # Disable the selected items
        self.image_processing_list.selectedItems()[0].setFlags(Qt.NoItemFlags)
        self.geometry_list.selectedItems()[0].setFlags(Qt.NoItemFlags)

    
        if key == 'led':
            self.mapping['led'][selected_image_proc_item] = selected_geometry_item
        elif key == 'camera':
            self.mapping['camera'] = {}
            self.mapping['camera'][selected_image_proc_item] = selected_geometry_item
        elif key == 'pupil':
            self.mapping['pupil'] = selected_geometry_item
        
    def execute(self):
        centered_geometry = self.get_centered_geometry()
        basis = utils.get_basis(centered_geometry['camera'])

        with h5.File(self.coord_filename, 'r') as f:
            frame_idxs = f['frame_idxs'][:]  
            sidx = np.argsort(frame_idxs)
            _frame_idxs = frame_idxs[sidx]
            
            # Check if frame idxs are sequential
            if np.prod(np.diff(frame_idxs)) != 1:
                print('Error: Frame idxs are not sequential. Please check.')
                return
            
            led_pix_co = {}
            pup_pix_co = f['pup_co'][:][sidx]

            temp = list(self.mapping['camera'].keys())[0]
            cam_pix_co = f['fiducial_coordinates'][temp][:][sidx]

            for k,v in f['fiducial_coordinates'].items():
                if k in self.mapping['led'].keys():
                    led_pix_co[k] = v[:][sidx]

        index = 0
        self.ax.clear()

        self.ax.plot(cam_pix_co[index,0], cam_pix_co[index,1], 'o', color='blue')
        self.ax.plot(pup_pix_co[index,0], pup_pix_co[index,1], 'o', color='green')

        for k,v in led_pix_co.items():
            self.ax.plot(v[index,0], v[index,1], 'o', color='red')

        self.ax.invert_yaxis()
        self.canvas.draw()

        # Find aberrations and correct
        camera_predictions = {}
        num_frames = cam_pix_co.shape[0]

        final_trajectories = {}
        
        for k,v in led_pix_co.items():
            dxdy = cam_pix_co-v
            median_offset = np.median(dxdy, axis=0)
            camera_predictions[k] = median_offset
        
            led_elevation , led_azimuth=  utils.get_led_angle(centered_geometry['leds'][k], basis)
            print(k, led_elevation, np.rad2deg(led_elevation), led_azimuth, np.rad2deg(led_azimuth))

            led_thetas = []
            led_phis = []

            for frame in tqdm.tqdm(range(num_frames)):

                led_pix = v[frame]
                pup_pix = pup_pix_co[frame]
                cam_pix = led_pix + camera_predictions[k]
                
                # Need a grid search here
                cam_pix[1] -= int(20)

                e,a = utils.calc_gaze_angle(pup_pix, led_pix, cam_pix, [led_elevation, led_azimuth])
                t,p = utils.ray_trace(centered_geometry, a, e) 

                led_thetas.append(np.pi/2 - t)
                led_phis.append(p)
            led_thetas = np.array(led_thetas)
            led_phis = np.array(led_phis)

            final_trajectories[k] = np.concatenate([led_thetas[:,None], led_phis[:,None]], axis=1)

        with h5.File(self.coord_filename, 'a') as h5_file:
            if 'raw_trajectories' in h5_file.keys():
                del h5_file['raw_trajectories']
            h5_file.create_group('raw_trajectories')
            for k,v in final_trajectories.items():
                h5_file['raw_trajectories'][k] = v

        # Quality control


            
    def get_centered_geometry(self):
        observer = self.mapping['pupil']
        observer = self.geometry_data[observer]
        observer_geometry= np.array([observer['x'], observer['y'], observer['z']])

    
        temp = list(self.mapping['camera'].keys())
        assert len(temp) == 1
        camera = self.mapping['camera'][temp[0]]
        camera = self.geometry_data[camera]
        camera_geometry = np.array([camera['x'], camera['y'], camera['z']])

        leds = list(self.mapping['led'].keys())
        led_geometry = {}
        for led_name in leds:
            led = self.mapping['led'][led_name]
            led = self.geometry_data[led]
            led = [led['x'], led['y'], led['z']]
            led_geometry[led_name] = np.array(led)


        centered_coordinates = {}
        centered_coordinates['observer'] = observer_geometry - observer_geometry
        centered_coordinates['camera'] = camera_geometry - observer_geometry
        centered_coordinates['leds'] = {}

        for led_name in leds:
            centered_coordinates['leds'][led_name] = led_geometry[led_name] - observer_geometry

        return centered_coordinates
