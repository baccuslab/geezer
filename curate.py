
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
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt

from utils import process_frame
import utils 

from geometry import GeometryTab

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


class CurateTab(QWidget):
    def __init__(self, main_window, parent=None):
        super(CurateTab, self).__init__(parent)
        # set icon
        self.main_window = main_window
        
        self.geezer_filename = None
        self.video = None
        self.frame_count = 0
        self.current_frame = 0
        self.view=0

        self.figure = Figure(tight_layout=True)
        self.axis = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('button_press_event', self.onclick)

        self.trace_figure = Figure(tight_layout=True)
        self.trace_axis = self.trace_figure.add_subplot(111)
        self.trace_canvas = FigureCanvas(self.trace_figure)
        self.trace_canvas.mpl_connect('button_press_event', self.onclick_trace)
        self.trace_axis.plot(np.arange(0,100))

        self.frame_slider = QSlider(Qt.Horizontal, self)
        self.frame_slider.valueChanged.connect(self.slider_changed)

        self.frame_enter = QLineEdit()
        self.frame_enter.setPlaceholderText('Frame')

        self.low_clim_slider = QSlider(Qt.Vertical, self)
        self.high_clim_slider = QSlider(Qt.Vertical, self)

        self.open_mp4_button = QPushButton('Load .mp4 file', self)
        self.open_geezer_button = QPushButton('Load geezer (centroids) output file', self)

        self.frame_slider.valueChanged.connect(self.slider_changed)
        self.low_clim_slider.valueChanged.connect(self.clim_changed)
        self.high_clim_slider.valueChanged.connect(self.clim_changed)
        self.frame_enter.returnPressed.connect(self.slider_changed)

        self.open_mp4_button.clicked.connect(self.open_video)
        self.open_geezer_button.clicked.connect(self.open_geezer)

        self.start_frame_process_edit = QLineEdit()
        self.end_frame_process_edit = QLineEdit()
        self.num_processes_edit = QLineEdit()
        
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

        self.low_clim_slider.setRange(0, 255)
        self.high_clim_slider.setRange(0, 255)
        self.low_clim_slider.setValue(0)
        self.high_clim_slider.setValue(255)

        self.image_layout = QHBoxLayout()
        self.image_layout.addWidget(self.canvas)
        self.image_layout.addWidget(self.low_clim_slider)
        self.image_layout.addWidget(self.high_clim_slider)

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
        
        layout = QVBoxLayout(self)
        
        box = QHBoxLayout()
        box.addWidget(self.open_mp4_button)
        box.addWidget(self.open_geezer_button)

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
        layout.addWidget(self.trace_canvas)

        self.setLayout(layout)

        
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

    def open_geezer(self):
        self.geezer_filename, _ = QFileDialog.getOpenFileName(self, 'Open Geezer File', ".", "Geezer Files (*.h5)")

        with h5.File(self.geezer_filename, 'r') as geezer_file:
            self.frame_idxs = geezer_file['frame_idxs'][:]
            self.order = self.frame_idxs.argsort()

            self.fiducials = {}
            for k,v in geezer_file['fids_co'].items():
                self.fiducials[k] = v[:][self.order]

            self.pup_co = geezer_file['pup_co'][:][self.order]




    def slider_changed(self):
        if self.current_frame == self.frame_slider.value():
            self.current_frame = int(float(self.frame_enter.text()))
        else:
            self.current_frame = self.frame_slider.value()
        self.update_frame()

    def clim_changed(self):
        self.update_frame()
    
    def onclick_trace(self, event):
        pass

    def onclick(self, event):
        self.last_co = [event.xdata, event.ydata]

    def update_frame(self):
        if self.video is not None:
            self.frame_enter.setText(str(self.current_frame))
            self.frame_slider.setValue(self.current_frame)
            
            if self.video.get(cv2.CAP_PROP_POS_FRAMES) != self.current_frame:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

            self.current_frame = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))

            ret, frame = self.video.read()
            frame = frame.mean(axis=2)
            
            # pp, fp = self.get_params()
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
                    #new
                    low = self.low_clim_slider.value()
                    high = self.high_clim_slider.value()
                    self.axis.imshow(frame, 'gray', clim=(low,high))

        # if self.geezer_filename is not None:
        #     if self.current_frame not in self.order:
        #         pup_co = self.pup_co[self.order[-1],:]
        #     else:
        #         self.update_frame()
        #         pup_co = self.pup_co[self.current_frame,:]
        #     for fid in self.fiducials:
        #         fid_co = self.fiducials[fid][self.current_frame,:]
        #         print(fid_co)
        #         self.axis.plot(fid_co[0], fid_co[1], 'co', alpha=0.5)
        #     self.axis.plot(pup_co[0], pup_co[1], 'ro', alpha=0.5)
            if self.geezer_filename is None:
                self.open_geezer()
                # if self.current_frame not in self.order:
                #     pup_co = self.pup_co[self.order[-1],:]
                # else:
                #     self.update_frame()
                #     pup_co = self.pup_co[self.current_frame,:]
                # self.axis.plot(pup_co[0], pup_co[1], 'ro', alpha=0.5)
                # if len(self.fiducials) > 0:
                #     for fid in self.fiducials:
                #         fid_co = self.fiducials[fid][self.current_frame,:]
                #         self.axis.plot(fid_co[0], fid_co[1], 'co', alpha=0.5)
            else:
                if self.current_frame in self.order:
                    self.axis.plot(self.pup_co[0], self.pup_co[1], 'ro')
                    for k, v in self.fiducials.items():
                        fid_co = v[self.order == self.current_frame][0]
                        self.axis.plot(fid_co[0], fid_co[1], 'co', alpha=0.5)

                else:
                    print(self.current_frame)
                    print(self.frame_idxs)

        self.canvas.draw()
        self.trace_canvas.draw()
        self.setFocus()

    def adjust_clim(self, image, low, high):
        # Adjust the contrast limits of the image here. This is a simple linear rescale, but you could replace this
        # with any function you want.
        image = np.clip(image, low, high)
        return 255 * (image - low) / (high - low)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Up:
            self.current_frame +=10
        elif event.key() == Qt.Key_Down:
            self.current_frame -=10
        elif event.key() == Qt.Key_Right:
            self.current_frame +=1
        elif event.key() == Qt.Key_Left:
            self.current_frame -=1
        

        self.update_frame()

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


    def process_video(self):

        # Make this run as a separate QThread
        # Popup window to get the filename of the h5 file where the data will be stored
        self.h5_filename = QFileDialog.getSaveFileName(self, 'Save File', '', 'HDF5 (*.h5)')[0]

        if self.h5_filename == '':
            return

        _video = cv2.VideoCapture(self.mp4_filename)
        meta = {} 
        start_frame = int(self.start_frame_process_edit.text())
        meta['start_frame'] = start_frame
        self.current_frame = start_frame
        self.frame_slider.setValue(start_frame)
        
        _video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = _video.read()

        proc_pup_params, proc_fid_params = self.get_params() 
        meta['pup_params'] = proc_pup_params
        meta['fid_params'] = proc_fid_params
        # p,f = process_frame(frame, self.pupil_co, self.fids_co, proc_pup_params, proc_fid_params)
        # print(p)
        # print(f)
        # self.pupil_co = p
        # self.fids_co = f

        self.update_frame()

        # Dialog window to ask if the user wants to save the results
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setText("Continue?")
        msg.setWindowTitle("Continue?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        retval = msg.exec_()
        if retval == QMessageBox.Yes:
            pass
        else:
            return


        num_processes = int(self.num_processes_edit.text())
        end_frame = int(self.end_frame_process_edit.text())
        meta['end_frame'] = end_frame
        
        total_frames = end_frame - start_frame
         
        # Calculate frames per process
        frames_per_process = total_frames // num_processes
        
        # Create a shared list to store the results
        result_list = mp.Manager().list()

        
        # Define a worker function for each process
        def worker(start_frame, end_frame, pxy, fxys):
            video = cv2.VideoCapture(self.mp4_filename)
            print('Started worker')
            local_results = []
            for frame_idx in tqdm.tqdm(range(start_frame, end_frame)):
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = video.read()
                frame = frame.mean(axis=2)
                if ret:
                    _processed_frame = process_frame(frame, pxy, fxys, proc_pup_params, proc_fid_params)
                    processed_frame = [frame_idx, _processed_frame]
                    
                    local_results.append(processed_frame)
            
            # Append local results to the shared list
            result_list.extend(local_results)
        
        # Create and start the processes
        processes = []
        for i in range(num_processes):
            proc_start_frame = i * frames_per_process + start_frame
            proc_end_frame = (i + 1) * frames_per_process + start_frame if i < num_processes - 1 else end_frame 
            process = mp.Process(target=worker, args=(proc_start_frame, proc_end_frame, self.pupil_co, self.fids_co))
            process.start()
            processes.append(process)
        
        # Wait for all processes to finish
        for process in processes:
            process.join()
        
        # Convert the shared list to a regular list
        results = list(result_list)

        print(len(results))
        save_frame_idxs = [x[0] for x in results] 
        save_pupil_co = [x[1][0] for x in results]
        save_fids_co = {} 
        num_fids = len(results[0][1][1])
        for i in range(num_fids):
            save_fids_co[i] = [x[1][1][i] for x in results] 
        
        with h5.File(self.h5_filename, 'w') as f:
            f.create_dataset('frame_idxs', data=save_frame_idxs)
            f.create_dataset('pup_co', data=save_pupil_co)
            f.create_group('fids_co')
            for i in range(num_fids):
                f.create_dataset('fids_co/fid_{}'.format(i), data=save_fids_co[i])
            f.create_dataset('meta', data=json.dumps(meta))

        
        # Close the video file
        _video.release()

