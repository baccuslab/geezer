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

DEFAULT_PARAMS = {'pupil': {'exp': 3, 'small': 20, 'large': 50, 'thresh': 200},
                  'fids': {'exp': 1, 'small': 3, 'large': 11, 'thresh': 65}}

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        # set icon
        self.setWindowIcon(QIcon('logo.png'))

        self.video = None
        self.frame_count = 0
        self.current_frame = 0
        self.view=0
        self.elevation = None
        self.azimuth = None
        self.line2 = None
        self.line3 = None
        self.marker2 = None
        self.marker3 = None
        
        self.pupil_co = None
        self.fids_co = []
        
        self.coordinate_file = None
        self.upper_left_crop_coords = None
        self.lower_right_crop_coords = None

        self.figure = Figure(tight_layout=True)
        self.axis = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('button_press_event', self.onclick)

        self.figure2 = Figure(tight_layout=True)
        self.axis2 = self.figure2.add_subplot(111)
        self.canvas2 = FigureCanvas(self.figure2)
        self.canvas2.mpl_connect('button_press_event', self.onclick)

        self.figure3 = Figure(tight_layout=True)
        self.axis3 = self.figure3.add_subplot(111)
        self.canvas3 = FigureCanvas(self.figure3)
        self.canvas3.mpl_connect('button_press_event', self.onclick)

        self.frame_slider = QSlider(Qt.Horizontal, self)
        self.frame_enter = QLineEdit()
        self.frame_enter.setPlaceholderText('Frame')
        self.frame_enter.returnPressed.connect(self.slider_changed)
        self.frame_slider.valueChanged.connect(self.slider_changed)


        self.low_clim_slider = QSlider(Qt.Vertical, self)
        self.high_clim_slider = QSlider(Qt.Vertical, self)

        self.open_button = QPushButton('Load .mp4 file', self)
        self.crop_button = QPushButton('Crop', self)
        self.load_button = QPushButton('Load coordinates', self)

        self.process_button = QPushButton('Process section', self)
        #self.plot_deg_button('Plot azimuth and elevation',self)#new
        self.load_pupil_button = QPushButton('Load pupil coordinates', self)
        self.plot_az_button = QPushButton('Plot azimuth and elevation', self)
        #self.plot_elev_button = QPushButton('Plot elevation', self)

        self.frame_slider.valueChanged.connect(self.slider_changed)
        self.low_clim_slider.valueChanged.connect(self.clim_changed)
        self.high_clim_slider.valueChanged.connect(self.clim_changed)
        self.open_button.clicked.connect(self.open_video)
        self.process_button.clicked.connect(self.process_video)
        self.load_pupil_button.clicked.connect(self.load_pupil_coordinates)
        self.plot_az_button.clicked.connect(self.plot_both)
        #self.plot_elev_button.clicked.connect(self.plot_elevation)
        #self.plot_deg_button.clicked.connect(self.plot_deg_button)#new

        # self.start_end_button.clicked.connect(self.start_end)
        self.load_button.clicked.connect(self.load_coordinates)

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

        self.crop_button.clicked.connect(self.crop_video)

            
        self.start_frame_process_edit.setPlaceholderText('0')
        self.end_frame_process_edit.setPlaceholderText('100')
        self.num_processes_edit.setPlaceholderText('4')

        self.start_frame_label = QLabel('Start frame')
        self.end_frame_label = QLabel('End frame')
        self.num_processes_label = QLabel('Number of processes')

        self.low_clim_slider.setRange(0, 255)
        self.high_clim_slider.setRange(0, 255)
        self.low_clim_slider.setValue(0)
        self.high_clim_slider.setValue(255)

        self.image_layout = QHBoxLayout()
        self.image_layout.addWidget(self.canvas)
        self.image_layout.addWidget(self.low_clim_slider)
        self.image_layout.addWidget(self.high_clim_slider)

        self.plot_az_layout = QHBoxLayout()
        self.plot_az_layout.addWidget(self.canvas2)
        #self.plot_az_layout.addWidget(self.plot_az_button)
        #self.plot_az_layout.addWidget(self.low_clim_slider)
        #self.plot_az_layout.addWidget(self.high_clim_slider)

        self.plot_elev_layout = QHBoxLayout()
        self.plot_elev_layout.addWidget(self.canvas3)

        self.plot_buttons_layout = QHBoxLayout()
        self.plot_buttons_layout.addWidget(self.load_pupil_button)
        #self.plot_buttons_layout.addWidget(self.plot_elev_button)
        self.plot_buttons_layout.addWidget(self.plot_az_button)
        #self.plot_elev_layout.addWidget(self.low_clim_slider)
        #self.plot_elev_layout.addWidget(self.high_clim_slider)
        
        self.plot_total = QHBoxLayout()
        self.plot_total.addLayout(self.plot_az_layout)
        self.plot_total.addLayout(self.plot_elev_layout)

        self.buttons_and_plot = QVBoxLayout()
        self.buttons_and_plot.addLayout(self.plot_total)
        self.buttons_and_plot.addLayout(self.plot_buttons_layout)

        self.mp4_and_plot_layout = QHBoxLayout()
        self.mp4_and_plot_layout.addLayout(self.image_layout)
        self.mp4_and_plot_layout.addLayout(self.buttons_and_plot)
        #self.mp4_and_plot_layout.addLayout(self.plot_az_layout)
        #self.mp4_and_plot_layout.addLayout(self.plot_elev_layout)
        #self.mp4_and_plot_layout.addLayout(self.plot_buttons_layout)

        self.process_layout = QHBoxLayout()
        self.process_layout.addWidget(self.start_frame_label)
        self.process_layout.addWidget(self.start_frame_process_edit)
        self.process_layout.addWidget(self.num_processes_label)
        self.process_layout.addWidget(self.num_processes_edit)
        self.process_layout.addWidget(self.end_frame_label)
        self.process_layout.addWidget(self.end_frame_process_edit)

        self.all_frames_checkbox = QCheckBox('All frames')
        self.all_frames_checkbox.setChecked(True)
        self.all_frames_checkbox.stateChanged.connect(self.select_all_frames)

        self.process_layout.addWidget(self.all_frames_checkbox)

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
        box.addWidget(self.open_button)
        box.addWidget(self.crop_button)

        layout.addLayout(box)

        layout.addLayout(self.pup_params_layout)
        layout.addLayout(self.fid_params_layout)
        layout.addLayout(self.mp4_and_plot_layout)#image_layout)

        layout.addLayout(self.frame_nav_layout)
        layout.addLayout(self.process_layout)

        layout.addWidget(self.process_button)
        # layout.addWidget(self.start_end_button)
        # Add a divider line
        self.divider_line = QFrame()
        self.divider_line.setFrameShape(QFrame.HLine)
        self.divider_line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(self.divider_line)

        layout.addWidget(self.load_button)
        self.setLayout(layout)

        self.load_params(DEFAULT_PARAMS)
    
    def load_params(self, params):
        self.pup_exp.setText(str(params['pupil']['exp']))
        self.pup_min.setText(str(params['pupil']['small']))
        self.pup_max.setText(str(params['pupil']['large']))
        self.pup_thresh.setText(str(params['pupil']['thresh']))

        self.fid_exp.setText(str(params['fids']['exp']))
        self.fid_min.setText(str(params['fids']['small']))
        self.fid_max.setText(str(params['fids']['large']))
        self.fid_thresh.setText(str(params['fids']['thresh']))



    def crop_video(self):

        ul = self.upper_left_crop_coords
        lr = self.lower_right_crop_coords

        x_start = ul[0]
        x_end = lr[0]

        y_start = ul[1]
        y_end = lr[1]

        # get a filepath from a qfiledialog
        crop_filename , _ = QFileDialog.getSaveFileName(self, 'Save file', ".", "mp4 files (*.mp4)")

        if crop_filename[-4:] != '.mp4':
            crop_filename += '.mp4'
        
        print(self.mp4_filename)
        # run the following command in a separate process
        command = 'ffmpeg -i {} -filter:v "crop={}:{}:{}:{}" {}'.format(self.mp4_filename, x_end-x_start, y_end-y_start, x_start, y_start, crop_filename)

        print(command)

        os.system(command)
        

    def load_coordinates(self):
        self.coordinate_file, _ = QFileDialog.getOpenFileName(self, 'Save file', ".", "h5 files (*.h5)")
        
        with h5.File(self.coordinate_file, 'r') as f: 
            self.coordinates = {}
            self.coordinates['frame_idxs'] = f['frame_idxs'][:]
            self.coordinates['pup_co'] = f['pup_co'][:]
        
            fids = f['fids_co']
            num_fids = len(list(fids.keys()))
            fid_keys = list(fids.keys())
            print(num_fids)
            self.coordinates['fiducials'] = {}
            for i in range(num_fids):
                self.coordinates['fiducials'][i] = fids[fid_keys[i]][:]


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



    def set_pupil_co(self):
        self.pupil_co = self.last_co
        self.update_frame()

    def add_fid_co(self):
        self.fids_co.append(self.last_co)
        self.update_frame()

    def slider_changed(self):
        if self.current_frame == self.frame_slider.value():
            self.current_frame = int(float(self.frame_enter.text()))
        else:
            self.current_frame = self.frame_slider.value()
        self.update_frame()
        self.plot_azimuth()
        self.plot_elevation()

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
                #new
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

    def adjust_clim(self, image, low, high):
        # Adjust the contrast limits of the image here. This is a simple linear rescale, but you could replace this
        # with any function you want.
        image = np.clip(image, low, high)
        return 255 * (image - low) / (high - low)

    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_1:
            self.view = 0
        elif event.key() == Qt.Key_2:
            self.view = 1
        elif event.key() == Qt.Key_3:
            self.view = 2
        elif event.key() == Qt.Key_4:
            self.view = 3
        elif event.key() == Qt.Key_5:
            self.view = 4
        elif event.key() == Qt.Key_Escape:
            self.pupil_co = None
            self.fids_co = []
        elif event.key() == Qt.Key_P:
            self.set_pupil_co()
        elif event.key() == Qt.Key_F:
            self.add_fid_co()
        elif event.key() == Qt.Key_Right:
            self.current_frame +=1
        elif event.key() == Qt.Key_Left:
            self.current_frame -=1

        elif event.key() == Qt.Key_L:
            self.upper_left_crop_coords = self.last_co
            print(self.last_co)

        elif event.key() == Qt.Key_R:
            self.lower_right_crop_coords = self.last_co
            print(self.last_co)

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
    
    def load_pupil_coordinates(self):
        #file_dialog = QFileDialog()
        #file_path, _ = file_dialog.getOpenFileName(self, 'Load Pupil Coordinates File', '', 'HDF5 Files (*.h5)')
        #if file_path:
        #    self.pupil_path = file_path
        self.load_coordinates()
        z = np.argsort(self.coordinates['frame_idxs'][:])
        self.azimuth = self.coordinates['pup_co'][:, 0][z]
        self.elevation = self.coordinates['pup_co'][:, 1][z]

    def plot_azimuth(self):
        self.axis2.clear()
        self.axis2.set_ylabel('Azimuth')
        self.axis2.set_xlabel('Time')
        self.line2, = self.axis2.plot(np.arange(len(self.azimuth)), self.azimuth, color='blue', label='Azimuth',zorder=1)
        if self.current_frame >= len(self.azimuth):
            self.current_frame = len(self.azimuth)
        self.marker2 = self.axis2.plot(self.current_frame, self.azimuth[self.current_frame], c='r', marker='o', label='Pupil Marker',zorder=2) #self.axis2.scatter(self.current_frame, self.azimuth[0], c='r', marker='o', label='Pupil Marker',zorder=2)
        self.axis2.legend()
        self.canvas2.draw()
        return

    def plot_elevation(self):
        self.axis3.clear()
        self.axis3.set_xlabel('Time')
        self.axis3.set_ylabel('Elevation')
        self.line3, = self.axis3.plot(np.arange(len(self.elevation)), self.elevation, color='blue', label='Elevation',zorder=1)
        if self.current_frame >= len(self.elevation):
            self.current_frame = len(self.elevation)
        self.marker3 = self.axis3.plot(self.current_frame, self.elevation[self.current_frame], c='r', marker='o', label='Pupil Marker',zorder=2) #self.axis3.scatter(self.current_frame, self.elevation[0], c='r', marker='o', label='Pupil Marker',zorder=2)
        self.axis3.legend()
        self.canvas3.draw()
        return

    def plot_both(self):
        self.plot_azimuth()
        self.plot_elevation()
        return

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


# import sys
# import cv2
# import numpy as np
# import h5py
# from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton, QSlider, QFileDialog
# from PyQt5.QtGui import QImage, QPixmap
# from PyQt5.QtCore import Qt, QSize
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.figure import Figure
# from matplotlib.animation import FuncAnimation

# class PupilTracker(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle('Pupil Tracker')
#         self.setGeometry(100, 100, 1200, 800)

#         self.video_path = None
#         self.frame_rate = None
#         self.frame_count = None
#         self.current_frame = 0

#         self.fiducials_path = None
#         self.pupil_path = None

#         self.azimuth = None
#         self.elevation = None

#         self.fig = Figure()
#         self.canvas = FigureCanvas(self.fig)
#         self.ax1 = self.fig.add_subplot(211)
#         self.ax2 = self.fig.add_subplot(212, sharex=self.ax1)
#         self.line1 = None
#         self.line2 = None
#         self.marker1 = None
#         self.marker2 = None

#         self.video_label = QLabel(self)
#         self.slider = QSlider(Qt.Horizontal, self)

#         self.setup_ui()

#     def setup_ui(self):
#         widget = QWidget(self)
#         layout = QVBoxLayout(widget)

#         load_video_button = QPushButton('Load Video')
#         load_video_button.clicked.connect(self.load_video)
#         layout.addWidget(load_video_button)

#         load_fiducials_button = QPushButton('Load Fiducials')
#         load_fiducials_button.clicked.connect(self.load_fiducials)
#         layout.addWidget(load_fiducials_button)

#         load_pupil_button = QPushButton('Load Pupil Coordinates')
#         load_pupil_button.clicked.connect(self.load_pupil_coordinates)
#         layout.addWidget(load_pupil_button)

#         layout.addWidget(self.video_label)

#         self.slider.setMinimum(0)
#         self.slider.setMaximum(0)
#         self.slider.valueChanged.connect(self.slider_value_changed)
#         layout.addWidget(self.slider)

#         layout.addWidget(self.canvas)
#         self.setCentralWidget(widget)

#     def load_video(self):
#         file_dialog = QFileDialog()
#         file_path, _ = file_dialog.getOpenFileName(self, 'Load Video File', '', 'MP4 Files (*.mp4)')
#         if file_path:
#             self.video_path = file_path
#             self.get_video_properties()
#             self.load_frame(0)

#     def get_video_properties(self):
#         cap = cv2.VideoCapture(self.video_path)
#         self.frame_rate = cap.get(cv2.CAP_PROP_FPS)
#         self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         cap.release()
#         self.slider.setMinimum(0)
#         self.slider.setMaximum(self.frame_count - 1)

#     def load_frame(self, frame_number):
#         cap = cv2.VideoCapture(self.video_path)
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
#         ret, frame = cap.read()
#         cap.release()
#         if ret:
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             height, width, channel = frame.shape
#             bytes_per_line = channel * width
#             q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
#             pixmap = QPixmap.fromImage(q_image)
#             self.video_label.setPixmap(pixmap.scaledToWidth(self.video_label.width()))

#     def load_fiducials(self):
#         file_dialog = QFileDialog()
#         file_path, _ = file_dialog.getOpenFileName(self, 'Load Fiducials File', '', 'HDF5 Files (*.h5)')
#         if file_path:
#             self.fiducials_path = file_path
#             self.load_coordinates()

#     def load_pupil_coordinates(self):
#         file_dialog = QFileDialog()
#         file_path, _ = file_dialog.getOpenFileName(self, 'Load Pupil Coordinates File', '', 'HDF5 Files (*.h5)')
#         if file_path:
#             self.pupil_path = file_path
#             self.load_coordinates()

#     def load_coordinates(self):
#         if self.fiducials_path is None or self.pupil_path is None:
#             return

#         with h5py.File(self.fiducials_path, 'r') as h5_file:
#             fiducials = h5_file['fids_co']
#             z = np.argsort(h5_file['frame_idxs'][:])

#         with h5py.File(self.pupil_path, 'r') as h5_file:
#             coordinates = h5_file['pup_co']
#             z = np.argsort(h5_file['frame_idxs'][:])
        
#         self.azimuth = coordinates[:, 0][z]
#         self.elevation = coordinates[:, 1][z]

#         self.ax1.clear()
#         self.ax1.set_ylabel('Azimuth')
#         self.ax2.clear()
#         self.ax2.set_xlabel('Time')
#         self.ax2.set_ylabel('Elevation')
#         self.line1, = self.ax1.plot(np.arange(len(self.azimuth)), self.azimuth, color='blue', label='Azimuth')
#         self.line2, = self.ax2.plot(np.arange(len(self.elevation)), self.elevation, color='green', label='Elevation')
#         self.marker1 = self.ax1.scatter(0, self.azimuth[0], c='r', marker='o', label='Pupil Marker')
#         self.marker2 = self.ax2.scatter(0, self.elevation[0], c='r', marker='o', label='Pupil Marker')
#         self.ax1.legend()
#         self.ax2.legend()
#         self.canvas.draw()

#     def slider_value_changed(self, value):
#         self.current_frame = value
#         self.load_frame(value)
#         self.update_coordinates(value)

#     def update_coordinates(self, frame):
#         self.marker1.set_offsets(np.array([[frame, self.azimuth[frame]]]))
#         self.marker2.set_offsets(np.array([[frame, self.elevation[frame]]]))
#         self.canvas.draw()

#     def resizeEvent(self, event):
#         self.update_plot_layout()
#         self.update_video_label_size()
#         super().resizeEvent(event)

#     def update_plot_layout(self):
#         self.fig.tight_layout()

#     def update_video_label_size(self):
#         pixmap = self.video_label.pixmap()
#         if pixmap is not None:
#             scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
#             self.video_label.setPixmap(scaled_pixmap)

#     def keyPressEvent(self, event):
#         if event.key() == Qt.Key_Escape:
#             self.close()

#     def show(self):
#         super().show()


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    # app2 = QApplication(sys.argv)
    # window2 = PupilTracker()
    # window2.show()
    window.show()
    app.exec_()
    # sys.exit(app2.exec_())