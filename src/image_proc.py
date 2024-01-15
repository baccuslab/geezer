from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QListWidget,
    QSlider,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QLineEdit,
    QMessageBox,
    QCheckBox,
    QFrame,
    QTabWidget,
    QMainWindow,
    QTableWidget,
    QHeaderView,
    QTableWidgetItem,
    QInputDialog,
)
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QDialog,
    QComboBox,
    QLineEdit,
    QHBoxLayout,
    QLabel,
)
import sys
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
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
from geometry import GeometryTab
import IPython

pupil_params = {}
## initialize the pupil contour parameters##
diagonal_size = 2**10

# boolean identity matrix
main_diagonal = np.eye(diagonal_size, diagonal_size, dtype=bool)

# setting up boolean matrices all False
half_diagonal = np.full((diagonal_size, diagonal_size), False, dtype=bool)
third_diagonal = half_diagonal.copy()
fourth_diagonal = half_diagonal.copy()
onethird = 1 / 3
onefourth = 1 / 4

invhalf_diagonal = half_diagonal.copy()
invthird_diagonal = third_diagonal.copy()
invfourth_diagonal = fourth_diagonal.copy()

# creating conditional boolean matrices
for i, _ in enumerate(half_diagonal):
    half_diagonal[int(i / 2), i] = True
    third_diagonal[int(i / 3), i] = True
    fourth_diagonal[int(i / 4), i] = True

    invfourth_diagonal[i, int(i / 2)] = True
    invthird_diagonal[i, int(i / 3)] = True
    invfourth_diagonal[i, int(i / 4)] = True

# vectors for radius storage
rr_stock = np.zeros((32), dtype=np.float64)
rr_2d = np.zeros((32, 2), dtype=np.float64)

rx_multiply = np.ones((32), dtype=np.float64)
ry_multiply = rx_multiply.copy()

crop_stock = np.zeros((32), dtype=int)

# constants that can be optimized based on the video input
onehalf_ry_add = [8, 10, 12, 14]
onehalf_rx_add = [8, 11, 12, 15]
onehalf_rx_subtract = [9, 10, 13, 14]
onehalf_ry_subtract = [9, 11, 13, 15]
onehalf_ry_multiplier = [8, 9, 10, 11]
onehalf_rx_multiplier = [12, 13, 14, 15]

onefourth_ry_add = [16, 19, 20, 21]
onefourth_rx_add = [16, 17, 20, 23]
onefourth_rx_subtract = [18, 19, 21, 22]
onefourth_ry_subtract = [17, 18, 22, 23]
onefourth_ry_multiplier = [16, 17, 18, 19]
onefourth_rx_multiplier = [20, 21, 22, 23]

onethird_ry_add = [24, 25, 28, 29]
onethird_rx_add = [24, 27, 28, 31]
onethird_rx_subtract = [25, 26, 29, 30]
onethird_ry_subtract = [26, 27, 30, 31]
onethird_ry_multiplier = [24, 25, 26, 27]
onethird_rx_multiplier = [28, 29, 30, 31]

# for multiplying operations
rx_multiplied = np.array(
    np.concatenate(
        (onehalf_rx_multiplier, onefourth_rx_multiplier, onethird_rx_multiplier)
    ),
    dtype=int,
)
ry_multiplied = np.array(
    np.concatenate(
        (onehalf_ry_multiplier, onefourth_ry_multiplier, onethird_ry_multiplier)
    ),
    dtype=int,
)
ones_ = np.ones(4, dtype=np.float64)
rx_multiply = np.array(
    np.concatenate((ones_ * 0.5, ones_ * onefourth, ones_ * onethird))
)
ry_multiply = np.array(
    np.concatenate((ones_ * 0.5, ones_ * onefourth, ones_ * onethird))
)

# for add operations
ry_add = np.array(
    np.concatenate(([0, 2, 4], onehalf_ry_add, onefourth_ry_add, onethird_ry_add)),
    dtype=int,
)
rx_add = np.array(
    np.concatenate(([1, 2, 5], onehalf_rx_add, onefourth_rx_add, onethird_ry_add)),
    dtype=int,
)

# for subtract oprtations
ry_subtract = np.array(
    np.concatenate(
        ([3, 5, 7], onehalf_ry_subtract, onefourth_ry_subtract, onethird_ry_subtract)
    )
)
rx_subtract = np.array(
    np.concatenate(
        ([3, 4, 6], onehalf_rx_subtract, onefourth_ry_subtract, onethird_rx_subtract)
    )
)

# pupil max and min radii
min_radius = 2
max_radius = 100


# rcParams: set default so that axis isn't shown and no x or yticks
plt.rcParams["axes.axisbelow"] = False
plt.rcParams["xtick.bottom"] = False
plt.rcParams["xtick.labelbottom"] = False
plt.rcParams["ytick.left"] = False
plt.rcParams["ytick.labelleft"] = False
plt.rcParams["axes.spines.left"] = False
plt.rcParams["axes.spines.bottom"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

DEFAULT_PARAMS = {
    "pupil": {"exp": 3, "small": 20, "large": 50, "thresh": 200},
    "fids": {"exp": 1, "small": 3, "large": 11, "thresh": 65},
}


class ImageProcTab(QWidget):
    def __init__(self, main_window, parent=None):
        super(ImageProcTab, self).__init__(parent)

        # set icon
        self.main_window = main_window
        self.setWindowIcon(QIcon("logo.png"))

        self.video = None
        self.dogs = None
        self.frame_count = 0
        self.current_frame = 0
        self.view = 0
        self.min_radius = min_radius
        self.max_radius = max_radius

        self.pupil_co = None
        self.fiducial_coordinates = {}
        self.selected_fiducial = None

        self.coordinate_file = None
        self.upper_left_crop_coords = None
        self.lower_right_crop_coords = None

        self.figure = Figure(tight_layout=True)
        self.axis = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect("button_press_event", self.onclick)

        self.frame_slider = QSlider(Qt.Horizontal, self)
        self.frame_enter = QLineEdit()
        self.frame_enter.setPlaceholderText("Frame")
        self.frame_enter.returnPressed.connect(self.slider_changed)
        self.frame_slider.valueChanged.connect(self.slider_changed)

        self.low_clim_slider = QSlider(Qt.Vertical, self)
        self.high_clim_slider = QSlider(Qt.Vertical, self)

        self.open_button = QPushButton("Load .mp4 file", self)
        self.crop_button = QPushButton("Crop", self)

        self.process_button = QPushButton("Process section", self)

        self.frame_slider.valueChanged.connect(self.slider_changed)
        self.low_clim_slider.valueChanged.connect(self.clim_changed)
        self.high_clim_slider.valueChanged.connect(self.clim_changed)
        self.open_button.clicked.connect(self.open_video)
        self.process_button.clicked.connect(self.process_video)
        # self.start_end_button.clicked.connect(self.start_end)

        self.start_frame_process_edit = QLineEdit()
        self.end_frame_process_edit = QLineEdit()
        self.num_processes_edit = QLineEdit()

        self.pup_params = []
        self.pup_labels = []
        self.pup_exp = QLineEdit()
        self.pup_exp_label = QLabel("Pup (power)")
        self.pup_params.append(self.pup_exp)
        self.pup_labels.append(self.pup_exp_label)

        self.pup_min = QLineEdit()
        self.pup_min_label = QLabel("Pup (small gauss)")
        self.pup_params.append(self.pup_min)
        self.pup_labels.append(self.pup_min_label)

        self.pup_max = QLineEdit()
        self.pup_max_label = QLabel("Pup (large gauss)")
        self.pup_params.append(self.pup_max)
        self.pup_labels.append(self.pup_max_label)

        self.pup_thresh = QLineEdit()
        self.pup_thresh_label = QLabel("Pup (binary thresh)")
        self.pup_params.append(self.pup_thresh)
        self.pup_labels.append(self.pup_thresh_label)

        self.fid_params = []
        self.fid_labels = []
        self.fid_exp = QLineEdit()
        self.fid_exp_label = QLabel("LED (power)")
        self.fid_params.append(self.fid_exp)
        self.fid_labels.append(self.fid_exp_label)

        self.fid_min = QLineEdit()
        self.fid_min_label = QLabel("LED (small gauss)")
        self.fid_params.append(self.fid_min)
        self.fid_labels.append(self.fid_min_label)

        self.fid_max = QLineEdit()
        self.fid_max_label = QLabel("LED (large gauss)")
        self.fid_params.append(self.fid_max)
        self.fid_labels.append(self.fid_max_label)

        self.fid_thresh = QLineEdit()
        self.fid_thresh_label = QLabel("LED (binary thresh)")
        self.fid_params.append(self.fid_thresh)
        self.fid_labels.append(self.fid_thresh_label)

        self.crop_button.clicked.connect(self.crop_video)

        self.start_frame_process_edit.setText("0")
        self.end_frame_process_edit.setText("100")
        self.num_processes_edit.setText("8")

        self.start_frame_label = QLabel("Start frame")
        self.end_frame_label = QLabel("End frame")
        self.num_processes_label = QLabel("Number of processes")

        self.low_clim_slider.setRange(0, 255)
        self.high_clim_slider.setRange(0, 255)
        self.low_clim_slider.setValue(0)
        self.high_clim_slider.setValue(255)

        self.fiducial_coordinate_list = QListWidget()
        self.fiducial_coordinate_list.itemClicked.connect(
            self.fiducial_coordinate_list_clicked)


        self.image_layout = QHBoxLayout()


        self.image_layout.addWidget(self.fiducial_coordinate_list)
        self.image_layout.addStretch(1)
        self.image_layout.addWidget(self.canvas)
        self.image_layout.addWidget(self.low_clim_slider)
        self.image_layout.addWidget(self.high_clim_slider)
        self.image_layout.addStretch(1)
        self.image_layout.addWidget(self.fiducial_coordinate_list)



        self.process_layout = QHBoxLayout()
        self.process_layout.addWidget(self.start_frame_label)
        self.process_layout.addWidget(self.start_frame_process_edit)
        self.process_layout.addWidget(self.num_processes_label)
        self.process_layout.addWidget(self.num_processes_edit)
        self.process_layout.addWidget(self.end_frame_label)
        self.process_layout.addWidget(self.end_frame_process_edit)

        self.all_frames_checkbox = QCheckBox("All frames")
        self.all_frames_checkbox.setChecked(False)
        self.all_frames_checkbox.stateChanged.connect(self.select_all_frames)

        self.process_layout.addWidget(self.all_frames_checkbox)

        self.pup_params_layout = QHBoxLayout()
        for label, param in zip(self.pup_labels, self.pup_params):
            self.pup_params_layout.addWidget(label)
            self.pup_params_layout.addWidget(param)

        self.fid_params_layout = QHBoxLayout()
        for label, param in zip(self.fid_labels, self.fid_params):
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
        layout.addLayout(self.image_layout)

        layout.addLayout(self.frame_nav_layout)
        layout.addLayout(self.process_layout)

        layout.addWidget(self.process_button)
        # layout.addWidget(self.start_end_button)
        # Add a divider line
        self.divider_line = QFrame()
        self.divider_line.setFrameShape(QFrame.HLine)
        self.divider_line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(self.divider_line)

        self.setLayout(layout)

        self.load_params(DEFAULT_PARAMS)

    def fiducial_coordinate_list_clicked(self, item):
        self.selected_fiducial = item.text()
        self.update_frame()

    def load_params(self, params):
        self.pup_exp.setText(str(params["pupil"]["exp"]))
        self.pup_min.setText(str(params["pupil"]["small"]))
        self.pup_max.setText(str(params["pupil"]["large"]))
        self.pup_thresh.setText(str(params["pupil"]["thresh"]))

        self.fid_exp.setText(str(params["fids"]["exp"]))
        self.fid_min.setText(str(params["fids"]["small"]))
        self.fid_max.setText(str(params["fids"]["large"]))
        self.fid_thresh.setText(str(params["fids"]["thresh"]))

    def crop_video(self):
        ul = self.upper_left_crop_coords
        lr = self.lower_right_crop_coords

        x_start = ul[0]
        x_end = lr[0]

        y_start = ul[1]
        y_end = lr[1]

        # get a filepath from a qfiledialog
        crop_filename, _ = QFileDialog.getSaveFileName(
            self, "Save file", ".", "mp4 files (*.mp4)"
        )

        if crop_filename[-4:] != ".mp4":
            crop_filename += ".mp4"

        print(self.mp4_filename)
        # run the following command in a separate process
        command = 'ffmpeg -i {} -filter:v "crop={}:{}:{}:{}" {}'.format(
            self.mp4_filename,
            x_end - x_start,
            y_end - y_start,
            x_start,
            y_start,
            crop_filename,
        )

        print(command)

        os.system(command)

    def open_video(self):
        self.mp4_filename, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", ".", "Video Files (*.mp4)"
        )

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
        # Popup to get name of fiducial
        fid_name, ok = QInputDialog.getText(
            self, "Input Dialog", "Enter fiducial name:"
        )

        if 'pup' in fid_name:
            print('Cannot use pup as a fiducial name')
            return

        self.fiducial_coordinates[fid_name] = self.last_co
        self.update_frame()

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
        if len(self.fiducial_coordinates.keys()) > 0:
            self.fiducial_coordinate_list.clear()

            for k, v in self.fiducial_coordinates.items():
                self.fiducial_coordinate_list.addItem("{}".format(k))



        if self.video is None:
            print('No video loaded')
            return

        self.frame_enter.setText(str(self.current_frame))
        self.frame_slider.setValue(self.current_frame)

        if self.video.get(cv2.CAP_PROP_POS_FRAMES) != self.current_frame:
            print("Setting frame")
            self.video.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

        ret, frame = self.video.read()
        print(self.current_frame, ret)
        frame = frame.mean(axis=2)

        pp, fp = self.get_params()
        if ret:
            if self.view == 0:
                frame = frame
                low = self.low_clim_slider.value()
                high = self.high_clim_slider.value()
                self.axis.clear()
                self.axis.imshow(frame, "gray", clim=(low, high))
            else:
                self.dogs = utils.dogs(frame, pp, fp)
                if self.view == 1:
                    frame = self.dogs[0]
                elif self.view == 2:
                    frame = self.dogs[1]
                elif self.view == 3:
                    frame = self.dogs[2]
                elif self.view == 4:
                    frame = self.dogs[3]

                self.axis.clear()
                low = self.low_clim_slider.value()
                high = self.high_clim_slider.value()
                self.axis.imshow(frame, "gray", clim=(low, high))

            if self.pupil_co is not None:
                self.axis.plot(self.pupil_co[0], self.pupil_co[1], "ro")
            if len(self.fiducial_coordinates.keys()) > 0:
                for k, v in self.fiducial_coordinates.items():
                    if k == self.selected_fiducial:
                        self.axis.plot(v[0], v[1], "ro")
                    else:
                        self.axis.plot(v[0], v[1], 'bo', alpha=0.7)

            if self.upper_left_crop_coords is not None:
                print("Plotting crop coords")
                self.axis.axvline(self.upper_left_crop_coords[0], color="r")
                self.axis.axhline(self.upper_left_crop_coords[1], color="r")
            if self.lower_right_crop_coords is not None:
                self.axis.axvline(self.lower_right_crop_coords[0], color="r")
                self.axis.axhline(self.lower_right_crop_coords[1], color="r")

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
            self.fiducial_coordinates = {}
        elif event.key() == Qt.Key_P:
            self.set_pupil_co()
        elif event.key() == Qt.Key_F:
            self.add_fid_co()
        elif event.key() == Qt.Key_Right:
            self.current_frame += 1
        elif event.key() == Qt.Key_Left:
            self.current_frame -= 1

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
            self.start_frame_process_edit.setText("0")
            self.end_frame_process_edit.setText(str(self.frame_count - 1))
            self.show()
        else:
            pass

    def process_video(self):
        # Make this run as a separate QThread
        # Popup window to get the filename of the h5 file where the data will be stored
        self.h5_filename = QFileDialog.getSaveFileName(
            self, "Save File", "", "HDF5 (*.h5)"
        )[0]

        if self.h5_filename == "":
            return

        _video = cv2.VideoCapture(self.mp4_filename)
        meta = {}
        start_frame = int(self.start_frame_process_edit.text())
        meta["start_frame"] = start_frame
        self.current_frame = start_frame
        self.frame_slider.setValue(start_frame)

        _video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = _video.read()

        proc_pup_params, proc_fid_params = self.get_params()
        meta["pup_params"] = proc_pup_params
        meta["fid_params"] = proc_fid_params

        # Check whether it found
        # REPLACE
        # Here
        self.update_frame()

        # Dialog window to ask if the user wants to save the results

        num_processes = int(self.num_processes_edit.text())
        end_frame = int(self.end_frame_process_edit.text())
        meta["end_frame"] = end_frame

        total_frames = end_frame - start_frame

        # Calculate frames per process
        frames_per_process = total_frames // num_processes

        # Create a shared list to store the results
        result_list = mp.Manager().list()

        # Define a worker function for each process
        def worker(start_frame, end_frame, pxy, fxys):
            video = cv2.VideoCapture(self.mp4_filename)
            print("Started worker")
            local_results = []
            for frame_idx in tqdm.tqdm(range(start_frame, end_frame)):
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = video.read()
                # frame = frame.mean(axis=2)
                if ret:
                    frame = frame.mean(axis=2)
                    _processed_frame = process_ellipse(
                        frame, pxy, fxys, proc_pup_params, proc_fid_params
                    )

                    processed_frame = [frame_idx, _processed_frame]
                    local_results.append(processed_frame)
                    # # pass

                else:
                    # input("Frame not found {}".format(frame_idx))
                    # print('Frame not found {}'.format(frame_idx))
                    processed_frame = [frame_idx, False]
                    local_results.append(processed_frame)

            # Append local results to the shared list
            result_list.extend(local_results)

        # Create and start the processes
        processes = []
        for i in range(num_processes):
            proc_start_frame = i * frames_per_process + start_frame
            proc_end_frame = (
                (i + 1) * frames_per_process + start_frame
                if i < num_processes - 1
                else end_frame
            )
            process = mp.Process(
                target=worker,
                args=(proc_start_frame, proc_end_frame, self.pupil_co, self.fiducial_coordinates),
            )
            process.start()
            processes.append(process)

        # Wait for all processes to finish
        for process in processes:
            process.join()

        # Convert the shared list to a regular list
        results = list(result_list)


        from IPython import embed
        embed()

        print(len(results))
        save_frame_idxs = np.array([x[0] for x in results])
        save_pupil_co = np.array([x[1][0] for x in results])
        save_widths = np.array([x[1][2] for x in results])
        save_heights = np.array([x[1][3] for x in results])
        save_phis = np.array([x[1][4] for x in results])
        save_fiducial_coordinates = {}
        

        fiducials_names = results[0][1][1].keys()
        for fiducial_name in fiducials_names:
            save_fiducial_coordinates[fiducial_name] = np.array([x[1][1][fiducial_name] for x in results])
    

        with h5.File(self.h5_filename, "w") as f:
            sidx = np.argsort(save_frame_idxs)
            frame_idxs = save_frame_idxs[sidx]

            f.create_dataset("frame_idxs", data=frame_idxs)

            save_pupil_co = np.array(save_pupil_co)[sidx]
            f.create_dataset("pup_co", data=save_pupil_co)

            f.create_dataset("width", data=save_widths[sidx])
            f.create_dataset("height", data=save_heights[sidx])
            f.create_dataset("phi", data=save_phis[sidx])

            f.create_group("fiducial_coordinates")
            for fiducial_name in fiducials_names:
                f.create_dataset("fiducial_coordinates/{}".format(fiducial_name), data=save_fiducial_coordinates[fiducial_name][sidx])

            f.create_dataset("meta", data=json.dumps(meta))

        # Close the video file
        _video.release()
