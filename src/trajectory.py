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
        self.map = {'observer': '', 'camera': '', 'leds': []}
        
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

        # Columns for Observer, Camera, and LEDs
        columns = ['Observer', 'Camera', 'LEDs']
        self.choice_layout = QHBoxLayout()
        for column in columns:
            self.create_column(self.choice_layout, column)
        
        layout.addLayout(self.choice_layout)
        
        bottom_row = QHBoxLayout()
        self.led_list = QListWidget()

        self.cam_y_offset_edit = QLineEdit()
        self.cam_y_offset_edit.setText('-8')

        self.execute_push_button = QPushButton('Execute')
        self.execute_push_button.clicked.connect(self.execute)
        
        bottom_row.addWidget(self.cam_y_offset_edit)
        bottom_row.addWidget(self.execute_push_button)
        bottom_row.setStretchFactor(self.execute_push_button, 1)
        bottom_row.addWidget(self.led_list)
        bottom_row.setStretchFactor(self.led_list, 1)

        layout.addLayout(bottom_row)
        # Create the matplotlib canvas for 3D plotting
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.ax = self.figure.add_subplot(111)

        self.figure.tight_layout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.save_button = QPushButton('Save phis and thetas')
        self.save_button.clicked.connect(self.save_phis_thetas)
        layout.addWidget(self.save_button)
        self.setLayout(layout)

        self.show()

    def execute(self):
        geometry_data = utils.load_geometry_json(self.geometry_filename)

        rig_geometry = {}
        rig_geometry['leds'] = []

        observer = self.map['observer']  
        observer = geometry_data[observer]
        observer = [observer['x'], observer['y'], observer['z']]

        camera = self.map['camera']  
        camera = geometry_data[camera]
        camera = [camera['x'], camera['y'], camera['z']]

        leds = self.map['leds']
        for led in leds:
            led = geometry_data[led]
            led = [led['x'], led['y'], led['z']]
            rig_geometry['leds'].append(led)

        rig_geometry['observer'] = observer
        rig_geometry['camera'] = camera

        centered_rig_geometry = {}
        centered_rig_geometry['leds'] = []

        for led in rig_geometry['leds']:
            centered_rig_geometry['leds'].append(np.array(led) - np.array(rig_geometry['observer']))
        centered_rig_geometry['observer'] = np.array(rig_geometry['observer']) - np.array(rig_geometry['observer'])
        centered_rig_geometry['camera'] = np.array(rig_geometry['camera']) - np.array(rig_geometry['observer'])

        basis = utils.get_basis(centered_rig_geometry)
        
        print(basis)

        with h5.File(self.coord_filename, 'r') as f:
            frame_idxs = f['frame_idxs'][:]  
            sidx = np.argsort(frame_idxs)
            frame_idxs = frame_idxs[sidx]
            
            led_co = [] 
            pup_co = f['pup_co'][:][sidx]
            for i,(k,v) in enumerate(f['fids_co'].items()):
                if i==0:
                    cam_co = v[:][sidx]
                else:
                    led_co.append(v[:][sidx])
        
        
        cam_preds = {}
        for led_idx in range(len(led_co)):
            dxdy = cam_co-led_co[led_idx]
            dxdy = np.mean(dxdy,axis=0)
            cam_preds[led_idx] = dxdy
        
        thetas = [] 
        phis = [] 
        for led_idx in range(len(led_co)):
            el, az =  utils.get_led_angle(centered_rig_geometry['leds'][led_idx], basis)
            
            _thetas = []
            _phis = []
            
            num_frames = frame_idxs.shape[0]
            for frame in tqdm.tqdm(range(num_frames)):
                idx = frame_idxs[frame]

                led_pix = led_co[led_idx][idx]
                pup_pix = pup_co[idx]
                cam_pix = led_co[led_idx][idx] + cam_preds[led_idx]
                cam_pix[1] -= int(self.cam_y_offset_edit.text())

                e,a = utils.calc_gaze_angle(pup_pix, led_pix, cam_pix, [el, az])
                t,p = utils.ray_trace(centered_rig_geometry, a, e) 
                _thetas.append(np.pi/2 - t)
                _phis.append(p)

            thetas.append(np.array(_thetas))
            phis.append(np.array(_phis))
        

        e = utils.euclidean_distance(led_co[0],led_co[3])
        e = np.abs(e - e[0])
        e = e >50 
        ee = utils.euclidean_distance(led_co[2],led_co[1])
        ee = np.abs(ee - ee[0])
        ee = ee >50 
        # plt.plot(e)
        # plt.show()
        # se = (thetas['se'])
        cleaned_up_phis = {}
        cleaned_up_thetas = {}
        for ii,led in enumerate(['ne','nw', 'se', 'sw']):
            try:
                sw = phis[ii] 
                sw = np.rad2deg(sw)
                sw = sw - sw[20000]
                # plt.plot(sw,'r')
                # plt.show()

                sw[e] = np.nan
                sw[ee] = np.nan

                # w = np.diff(sw) > 3 
                # w = np.concatenate([w,[False]])
                # sw[w] = np.nan
                # y = sw
                sw[e] = np.nan
                sw[ee] = np.nan
                nans, x = utils.nan_helper(sw)
                print('nans', nans.shape)
                # # interpolate with scipy

                # w = np.diff(sw) > 5
                # sw[:-1][w] = np.nan
                # nans, x = utils.nan_helper(sw)
                sw[nans]= np.interp(x(nans), x(~nans), sw[~nans])
                # plt.plot(sw,'r')
                # plt.show()

                w = np.abs(np.diff(sw)) > 1 
                sw[1:][w] = np.nan

                sw[sw > 20] = np.nan
                sw[sw < -20] = np.nan
                nans, x = utils.nan_helper(sw)
                sw[nans]= np.interp(x(nans), x(~nans), sw[~nans])
                # plt.plot(sw, 'k', lw=0.5)
                # plt.show()
                cleaned_up_phis[led] = sw - sw[0]

                sw = thetas[ii] 
                sw = np.rad2deg(sw)
                sw = sw - sw[20000]

                sw[e] = np.nan
                sw[ee] = np.nan

                sw[e] = np.nan
                sw[ee] = np.nan
                nans, x = utils.nan_helper(sw)
                # # interpolate with scipy

                # w = np.diff(sw) > 5
                # sw[:-1][w] = np.nan
                # nans, x = utils.nan_helper(sw)
                sw[nans]= np.interp(x(nans), x(~nans), sw[~nans])
                # plt.plot(sw,'r')
                # plt.show()

                w = np.abs(np.diff(sw)) > 1 
                sw[1:][w] = np.nan

                sw[sw > 5] = np.nan
                sw[sw < -5] = np.nan
                nans, x = utils.nan_helper(sw)
                print('nans', nans.shape)
                sw[nans]= np.interp(x(nans), x(~nans), sw[~nans])
                # plt.plot(sw, 'k', lw=0.5)
                # plt.show()
                cleaned_up_thetas[led] = sw - sw[0]

            except:
                cleaned_up_phis[led] = np.zeros_like(sw)
                cleaned_up_thetas[led] = np.zeros_like(sw)


            
        self.ax.clear()
        for led in cleaned_up_phis:
            self.ax.plot(cleaned_up_phis[led], label=led)
        self.ax.legend()
        self.canvas.draw()

        self.phis = cleaned_up_phis
        self.thetas = cleaned_up_thetas

    def create_column(self, layout, name):
        hbox = QVBoxLayout()
        combo_box = QComboBox()
        button = QPushButton(name.lower())
        button.clicked.connect(lambda: self.populate_map(name.lower(), combo_box.currentText()))
        hbox.addWidget(QLabel(name))
        hbox.addWidget(combo_box)
        hbox.addWidget(button)
        self.combo_boxes[name.lower()] = combo_box
        self.combo_buttons[name.lower()] = button

        layout.addLayout(hbox)
        layout.setStretchFactor(hbox, 1)

    def load_coords(self):
        # Popup window to select file
        self.coord_filename = QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        self.coord_label.setText(self.coord_filename)
        
    def populate_map(self, key, value):
        if key == 'leds':
            self.map[key].append(value)
            self.led_list.addItem(value)
        else:
            self.map[key] = value

            self.combo_boxes[key].setEnabled(False)
            self.combo_buttons[key].setEnabled(False)

    
    def update(self):
        if self.geometry_filename is not None:
            self.geo_label.setText(self.geometry_filename)
            self.geo_data= utils.load_geometry_json(self.geometry_filename)
            
            for k,v in self.combo_boxes.items():
                v.clear()
                v.addItems(list(self.geo_data.keys()))
                v.setCurrentIndex(0)

    def save_phis_thetas(self):
        # Popup for save path
        # Save phis and thetas
        with h5.File(self.coord_filename, 'a') as f:
            frame_idxs = f['frame_idxs'][:]  
            sidx = np.argsort(frame_idxs)
            frame_idxs = frame_idxs[sidx]
            
            led_co = [] 
            pup_co = f['pup_co'][:][sidx]
            for i,(k,v) in enumerate(f['fids_co'].items()):
                if i==0:
                    cam_co = v[:][sidx]
                else:
                    led_co.append(v[:][sidx])
        
        filename = QFileDialog.getSaveFileName(self, 'Save file', '/home')[0]
        if filename is not None:
            pickle.dump(self.phis, open(filename+'_phis.pkl','wb'))
            pickle.dump(self.thetas, open(filename+'_thetas.pkl','wb'))


            # # populate combo boxes with geometry_data[name]
            # self.observer_combo.clear()
            # self.cam_combo.clear()

