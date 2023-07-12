import sys
import cv2
import numpy as np
import h5py
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton, QSlider, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

from PyQt5.QtWidgets import QApplication, QWidget, QSlider, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QLineEdit, QMessageBox, QCheckBox, QFrame
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

class PupilTracker(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Pupil Tracker')
        self.setGeometry(100, 100, 1200, 800)

        self.video = None
        self.video_path = None
        self.frame_rate = None
        self.frame_count = 0
        self.current_frame = 0

        self.pupil_co = None #fromold 
        self.fids_co = []  #fromold

        self.coordinate_file = None #fromold

        self.fiducials_path = None
        self.pupil_path = None

        self.azimuth = None
        self.elevation = None

        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212, sharex=self.ax1)
        self.line1 = None
        self.line2 = None
        self.marker1 = None
        self.marker2 = None
        self.canvas.mpl_connect('button_press_event', self.onclick) #from old

        #all from old
        self.open_button = QPushButton('Load .mp4 file', self)
        self.frame_slider = QSlider(Qt.Horizontal, self)
        self.frame_enter = QLineEdit()
        self.frame_enter.setPlaceholderText('Frame')
        self.frame_enter.returnPressed.connect(self.slider_changed)
        self.frame_slider.valueChanged.connect(self.slider_changed)
        self.low_clim_slider = QSlider(Qt.Vertical, self)
        self.high_clim_slider = QSlider(Qt.Vertical, self)
        self.frame_slider.valueChanged.connect(self.slider_changed)
        self.low_clim_slider.valueChanged.connect(self.clim_changed)
        self.high_clim_slider.valueChanged.connect(self.clim_changed)
        self.open_button.clicked.connect(self.open_video)
        self.low_clim_slider.setRange(0, 255)
        self.high_clim_slider.setRange(0, 255)
        self.low_clim_slider.setValue(0)
        self.high_clim_slider.setValue(255)
        self.image_layout = QHBoxLayout()
        self.image_layout.addWidget(self.canvas)
        self.image_layout.addWidget(self.low_clim_slider)
        self.image_layout.addWidget(self.high_clim_slider)
        self.pup_params = []
        self.pup_labels = []
        self.pup_exp = QLineEdit()
        self.pup_exp_label = QLabel('Pup (power)')
        self.pup_params.append(self.pup_exp)
        self.pup_labels.append(self.pup_exp_label)

        self.pup_min = QLineEdit()
        self.pup_min_label = QLabel('Pup (small gauss)')
        self.pup_params.append(self.pup_min)
        self.pup_labels.append(self.pup_min_label)

        self.pup_max = QLineEdit()
        self.pup_max_label = QLabel('Pup (large gauss)')
        self.pup_params.append(self.pup_max)
        self.pup_labels.append(self.pup_max_label)

        self.pup_thresh = QLineEdit()
        self.pup_thresh_label = QLabel('Pup (binary thresh)')
        self.pup_params.append(self.pup_thresh)
        self.pup_labels.append(self.pup_thresh_label)

        self.fid_params = []
        self.fid_labels = []
        self.fid_exp = QLineEdit()
        self.fid_exp_label = QLabel('LED (power)') 
        self.fid_params.append(self.fid_exp)
        self.fid_labels.append(self.fid_exp_label)

        self.fid_min = QLineEdit()
        self.fid_min_label = QLabel('LED (small gauss)')
        self.fid_params.append(self.fid_min)
        self.fid_labels.append(self.fid_min_label)

        self.fid_max = QLineEdit()
        self.fid_max_label = QLabel('LED (large gauss)')
        self.fid_params.append(self.fid_max)
        self.fid_labels.append(self.fid_max_label)

        self.fid_thresh = QLineEdit()
        self.fid_thresh_label = QLabel('LED (binary thresh)')
        self.fid_params.append(self.fid_thresh)
        self.fid_labels.append(self.fid_thresh_label)

        self.pup_params_layout = QHBoxLayout()
        for label, param in zip(self.pup_labels,self.pup_params):
            self.pup_params_layout.addWidget(label)
            self.pup_params_layout.addWidget(param)

        self.fid_params_layout = QHBoxLayout()
        for label, param in zip(self.fid_labels,self.fid_params):
            self.fid_params_layout.addWidget(label)
            self.fid_params_layout.addWidget(param)

        self.frame_nav_layout = QHBoxLayout()
        self.frame_nav_layout.addWidget(self.frame_slider)
        self.frame_nav_layout.setStretch(0, 1)
        self.frame_nav_layout.addWidget(self.frame_enter)

        self.video_label = QLabel(self)
        self.slider = QSlider(Qt.Horizontal, self)

        self.setup_ui()

        

    def setup_ui(self):
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
       
        #all from old
        box = QHBoxLayout()
        box.addWidget(self.open_button)
        layout.addLayout(box)
        layout.addLayout(self.pup_params_layout)
        layout.addLayout(self.fid_params_layout)
        layout.addLayout(self.image_layout)
        layout.addLayout(self.frame_nav_layout)
        # layout.addWidget(self.start_end_button)
        # Add a divider line
        self.divider_line = QFrame()
        self.divider_line.setFrameShape(QFrame.HLine)
        self.divider_line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(self.divider_line)

        layout.addWidget(self.load_button)
        #self.setLayout(layout)


        load_video_button = QPushButton('Load Video')
        load_video_button.clicked.connect(self.load_video)
        layout.addWidget(load_video_button)

        load_fiducials_button = QPushButton('Load Fiducials')
        load_fiducials_button.clicked.connect(self.load_fiducials)
        layout.addWidget(load_fiducials_button)

        load_pupil_button = QPushButton('Load Pupil Coordinates')
        load_pupil_button.clicked.connect(self.load_pupil_coordinates)
        layout.addWidget(load_pupil_button)

        layout.addWidget(self.video_label)

        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self.slider_value_changed)
        layout.addWidget(self.slider)

        layout.addWidget(self.canvas)
        self.setCentralWidget(widget)

        self.load_params(DEFAULT_PARAMS)

        def get_params(self):
            pup_params = []
            for param in self.pup_params:
                pup_params.append(int(float(param.text())))
            
            fid_params = []
            for param in self.fid_params:
                fid_params.append(int(float(param.text())))
            # pup_params = self.pup_params_edit.text()

            # fid_params = self.fid_params_edit.text()
            # pup_params = pup_params.split(',')
            # fid_params = fid_params.split(',')
            

            # pup_params = [int(float(x)) for x in pup_params]
            # fid_params = [int(float(x)) for x in fid_params]

            return pup_params, fid_params
    
    def select_all_frames(self):
        all_frames = self.all_frames_checkbox.isChecked()

        if all_frames:
            self.start_frame_process_edit.setText('0')
            self.end_frame_process_edit.setText(str(self.frame_count - 1))
            self.show()
        else:
            pass

    def slider_changed(self):
        if self.current_frame == self.frame_slider.value():
            self.current_frame = int(float(self.frame_enter.text()))
        else:
            self.current_frame = self.frame_slider.value()
        self.update_frame()

    def clim_changed(self):
        self.update_frame()

    def onclick(self, event):
        self.last_co = [event.xdata, event.ydata]

    def update_frame(self):
        if self.video is None:
            return

        self.frame_enter.setText(str(self.current_frame))
        self.frame_slider.setValue(self.current_frame)
        
        if self.video.get(cv2.CAP_PROP_POS_FRAMES) != self.current_frame:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.video.read()
        frame = frame.mean(axis=2)
        
        pp, fp = self.get_params()
        if ret:
            if self.view == 0:
                frame = frame
                low = self.low_clim_slider.value()
                high = self.high_clim_slider.value()
                self.axis.clear()
                self.axis.imshow(frame, 'gray', clim=(low, high))
            else:
                dogs = utils.dogs(frame, pp, fp)
                if self.view == 1:
                    frame = dogs[0]
                elif self.view == 2:
                    frame = dogs[1]
                elif self.view == 3:
                    frame = dogs[2]
                elif self.view == 4:
                    frame = dogs[3]


                self.axis.clear()
                low = self.low_clim_slider.value()
                high = self.high_clim_slider.value()
                self.axis.imshow(frame, 'gray', clim=(low,high))
            
            if self.coordinate_file is None:
                if self.pupil_co is not None:
                    self.axis.plot(self.pupil_co[0], self.pupil_co[1], 'ro')
                if len(self.fids_co) > 0:
                    for fid in self.fids_co:
                        self.axis.plot(fid[0], fid[1], 'bo')
            else:
                if self.current_frame in self.coordinates['frame_idxs']:
                    self.pupil_co = self.coordinates['pup_co'][self.coordinates['frame_idxs'] == self.current_frame][0]
                    self.axis.plot(self.pupil_co[0], self.pupil_co[1], 'ro')
                    for k, v in self.coordinates['fiducials'].items():
                        fid_co = v[self.coordinates['frame_idxs'] == self.current_frame][0]
                        self.axis.plot(fid_co[0], fid_co[1], 'co', alpha=0.5)

                else:
                    print(self.current_frame)
                    print(self.coordinates['frame_idxs'])

            if self.upper_left_crop_coords is not None:
                print('Plotting crop coords')
                self.axis.axvline(self.upper_left_crop_coords[0], color='r')
                self.axis.axhline(self.upper_left_crop_coords[1], color='r')
            if self.lower_right_crop_coords is not None:
                self.axis.axvline(self.lower_right_crop_coords[0], color='r')
                self.axis.axhline(self.lower_right_crop_coords[1], color='r')

            self.canvas.draw()
            self.setFocus()

    def load_params(self, params):
        self.pup_exp.setText(str(params['pupil']['exp']))
        self.pup_min.setText(str(params['pupil']['small']))
        self.pup_max.setText(str(params['pupil']['large']))
        self.pup_thresh.setText(str(params['pupil']['thresh']))

        self.fid_exp.setText(str(params['fids']['exp']))
        self.fid_min.setText(str(params['fids']['small']))
        self.fid_max.setText(str(params['fids']['large']))
        self.fid_thresh.setText(str(params['fids']['thresh']))

    def load_video(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, 'Load Video File', '', 'MP4 Files (*.mp4)')
        if file_path:
            self.video_path = file_path
            self.get_video_properties()
            self.load_frame(0)

    def get_video_properties(self):
        cap = cv2.VideoCapture(self.video_path)
        self.frame_rate = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.frame_count - 1)
        ret, frame = self.video.read()
        self.low_clim_slider.setMaximum(np.max(frame))
        self.low_clim_slider.setValue(np.min(frame))
        self.high_clim_slider.setMaximum(np.max(frame))
        self.high_clim_slider.setValue(np.max(frame))

        self.update_frame()

    def load_frame(self, frame_number):
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytes_per_line = channel * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.video_label.setPixmap(pixmap.scaled(800, 600, Qt.KeepAspectRatio))

    def open_video(self):
        self.mp4_filename, _ = QFileDialog.getOpenFileName(self, 'Open Video File', ".", "Video Files (*.mp4)")

        if self.mp4_filename:

    
            self.video = cv2.VideoCapture(self.mp4_filename)
            self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_slider.setMaximum(self.frame_count - 1)
            self.frame_slider.setValue(0)
            ret, frame = self.video.read()
            self.low_clim_slider.setMaximum(np.max(frame))
            self.low_clim_slider.setValue(np.min(frame))
            self.high_clim_slider.setMaximum(np.max(frame))
            self.high_clim_slider.setValue(np.max(frame))


            self.update_frame()


    def load_fiducials(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, 'Load Fiducials File', '', 'HDF5 Files (*.h5)')
        if file_path:
            self.fiducials_path = file_path
            self.load_coordinates()

    def load_pupil_coordinates(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, 'Load Pupil Coordinates File', '', 'HDF5 Files (*.h5)')
        if file_path:
            self.pupil_path = file_path
            self.load_coordinates()

    def load_coordinates(self):
        if self.fiducials_path is None or self.pupil_path is None:
            return

        with h5py.File(self.fiducials_path, 'r') as h5_file:
            fiducials = h5_file['fiducials'][:]

        with h5py.File(self.pupil_path, 'r') as h5_file:
            coordinates = h5_file['coordinates'][:]

        self.azimuth = coordinates[0, :]
        self.elevation = coordinates[1, :]

        self.ax1.clear()
        self.ax1.set_ylabel('Azimuth')
        self.ax2.clear()
        self.ax2.set_xlabel('Time')
        self.ax2.set_ylabel('Elevation')
        self.line1, = self.ax1.plot(np.arange(len(self.azimuth)), self.azimuth, color='blue', label='Azimuth')
        self.line2, = self.ax2.plot(np.arange(len(self.elevation)), self.elevation, color='green', label='Elevation')
        self.marker1 = self.ax1.scatter(0, self.azimuth[0], c='r', marker='o', label='Pupil Marker')
        self.marker2 = self.ax2.scatter(0, self.elevation[0], c='r', marker='o', label='Pupil Marker')
        self.ax1.legend()
        self.ax2.legend()
        self.canvas.draw()

    def slider_value_changed(self, value):
        self.current_frame = value
        self.load_frame(value)
        self.update_coordinates(value)

    def update_coordinates(self, frame):
        self.marker1.set_offsets(np.array([[frame, self.azimuth[frame]]]))
        self.marker2.set_offsets(np.array([[frame, self.elevation[frame]]]))
        self.canvas.draw()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

    def show(self):
        super().show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PupilTracker()
    window.show()
    sys.exit(app.exec_())
