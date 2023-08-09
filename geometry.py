import json
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


class GeometryTab(QWidget):
    def __init__(self, main_window, parent=None):
        super(GeometryTab, self).__init__(parent)
          
        # Create the matplotlib canvas for 3D plotting
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')
        
        # Create the push buttons
        self.add_object_button = QPushButton("Add Object", self)
        self.add_object_button.clicked.connect(self.show_input_dialog)
        
        self.delete_button = QPushButton("Delete Selected", self)
        self.delete_button.clicked.connect(self.delete_selected_object)
        
        # Create the table
        self.table = QTableWidget(0, 5, self)  # 0 rows, 5 columns initially
        self.table.setHorizontalHeaderLabels(['Name', 'Type', 'x', 'y', 'z'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)  # Adjust column width to content
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)  # Make table un-editable
        
        # Set up the layout
        main_layout = QHBoxLayout(self)
        
        # Table and buttons on the left side
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.table)
        left_layout.addWidget(self.add_object_button)
        left_layout.addWidget(self.delete_button)

        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self.save_items)

        self.load_button = QPushButton("Load", self)
        self.load_button.clicked.connect(self.load_items)

        left_layout.addWidget(self.save_button)
        left_layout.addWidget(self.load_button)
        
        main_layout.addLayout(left_layout)
        main_layout.addWidget(self.canvas)
        
        # Set the stretch factors
        main_layout.setStretchFactor(left_layout, 1)  # table + buttons
        main_layout.setStretchFactor(self.canvas, 3)  # plot


        
        self.setLayout(main_layout)
        
        # To track plot items
        self.plot_items = {}
        
    def show_input_dialog(self):
        dialog = ObjectInputDialog(self)
        dialog.object_added.connect(self.add_and_plot_object)
        dialog.exec_()

    def delete_selected_object(self):
        current_row = self.table.currentRow()
        if current_row == -1:
            return  # No row is selected
        
        # Remove the plot item
        plot_item = self.plot_items.pop(current_row, None)
        if plot_item:
            plot_item.remove()
            self.canvas.draw()
        
        # Remove the row from the table
        self.table.removeRow(current_row)               # Create the matplotlib canvas for 3D plotting
    
    def add_and_plot_object(self, data):
        name = data["name"]
        object_type = data["type"]
        
        if object_type == "point":
            x, y, z = float(data["x"]), float(data["y"]), float(data["z"])
            plot_item = self.ax.scatter(x, y, z, c='r', marker='o')
            
            # Add to the table
            self.table.insertRow(self.table.rowCount())
            self.table.setItem(self.table.rowCount()-1, 0, QTableWidgetItem(name))
            self.table.setItem(self.table.rowCount()-1, 1, QTableWidgetItem(object_type))
            self.table.setItem(self.table.rowCount()-1, 2, QTableWidgetItem(str(x)))
            self.table.setItem(self.table.rowCount()-1, 3, QTableWidgetItem(str(y)))
            self.table.setItem(self.table.rowCount()-1, 4, QTableWidgetItem(str(z)))
            
        elif object_type == "plane":
            ll = float(data['x'][0]), float(data['y'][0]), float(data['z'][0])
            lr = float(data['x'][1]), float(data['y'][1]), float(data['z'][1])
            ul = float(data['x'][2]), float(data['y'][2]), float(data['z'][2])

            
            # Calculate upper right coordinate
            ur = [lr[0] + (ul[0] - ll[0]), lr[1] + (ul[1] - ll[1]), lr[2] + (ul[2] - ll[2])]
            
            # Plotting the plane
            x = [ll[0], lr[0], ur[0], ul[0], ll[0]]
            y = [ll[1], lr[1], ur[1], ul[1], ll[1]]
            z = [ll[2], lr[2], ur[2], ul[2], ll[2]]
            plot_item = self.ax.plot(x, y, z, color='black')
            
            # Add to the table (3 rows for 3 coordinates)
            self.table.insertRow(self.table.rowCount())
            self.table.setItem(self.table.rowCount()-1, 0, QTableWidgetItem(name))
            self.table.setItem(self.table.rowCount()-1, 1, QTableWidgetItem(object_type))
            
            xs = [ll[0], lr[0], ul[0]]
            ys = [ll[1], lr[1], ul[1]]
            zs = [ll[2], lr[2], ul[2]]
            

            self.table.setItem(self.table.rowCount()-1, 2, QTableWidgetItem(str(xs)))
            self.table.setItem(self.table.rowCount()-1, 3, QTableWidgetItem(str(ys)))
            self.table.setItem(self.table.rowCount()-1, 4, QTableWidgetItem(str(zs)))
            
        # Store the plot item in the dictionary with the row number as the key
        row_key = self.table.rowCount() - 1 if object_type == "point" else self.table.rowCount() - 3
        self.plot_items[row_key] = plot_item
        
        # Refresh the canvas
        self.canvas.draw()
    def save_items(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getSaveFileName(self, "Save File", "", "JSON Files (*.json);;All Files (*)", options=options)
        
        if not filePath:
            return
        
        data = []
        for row in range(self.table.rowCount()):
            item_data = {
                "name": self.table.item(row, 0).text(),
                "type": self.table.item(row, 1).text()
            }
        
            if item_data['type'] == 'point':
                item_data["x"] = float(self.table.item(row, 2).text())
                item_data["y"] = float(self.table.item(row, 3).text())
                item_data["z"] = float(self.table.item(row, 4).text())
            elif item_data['type'] == 'plane':
                item_data["x"] = [float(x) for x in self.table.item(row, 2).text()[1:-1].split(',')]
                item_data["y"] = [float(y) for y in self.table.item(row, 3).text()[1:-1].split(',')]
                item_data["z"] = [float(z) for z in self.table.item(row, 4).text()[1:-1].split(',')]


            data.append(item_data)
        with open(filePath, 'w') as file:
            json.dump(data, file)

    def load_items(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "Load File", "", "JSON Files (*.json);;All Files (*)", options=options)
        
        if not filePath:
            return
        
        with open(filePath, 'r') as file:
            data = json.load(file)
    

        self.table.setRowCount(0)  # Clear the table
        for plot_item in self.plot_items.values():
            plot_item.remove()

        self.plot_items.clear()  # Clear the plot items
        self.canvas.draw()
        
        for item_data in data:
            self.add_and_plot_object(item_data)

class ObjectInputDialog(QDialog):
    # Signal to emit when an object is ready to be added
    object_added = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Object")
        
        # Create the drop-down menu (QComboBox)
        self.object_type_combo = QComboBox(self)
        self.object_type_combo.addItems(["point", "plane"])
        self.object_type_combo.currentTextChanged.connect(self.on_object_type_changed)
        
        # Create the name input (QLineEdit)
        self.name_edit = QLineEdit(self)
        self.name_edit.setPlaceholderText("name")
        
        # Create x, y, z input fields for the point
        self.x_edit = QLineEdit(self)
        self.y_edit = QLineEdit(self)
        self.z_edit = QLineEdit(self)
        for edit in [self.x_edit, self.y_edit, self.z_edit]:
            edit.setPlaceholderText(edit.objectName().split("_")[0])  # x, y, or z
        
        # Create x, y, z input fields for the plane
        self.plane_coords = {}
        for label in ["lower left", "lower right", "upper left"]:
            self.plane_coords[label] = {
                "x": QLineEdit(self),
                "y": QLineEdit(self),
                "z": QLineEdit(self)
            }
            for key, edit in self.plane_coords[label].items():
                edit.setPlaceholderText(key)
        
        # Add to plot button
        self.add_button = QPushButton("Add to Plot", self)
        self.add_button.clicked.connect(self.add_object)
        
        # Layout setup for the dialog
        layout = QVBoxLayout(self)
        layout.addWidget(self.object_type_combo)
        layout.addWidget(self.name_edit)
        
        self.point_coords_layout = QHBoxLayout()
        self.point_coords_layout.addWidget(QLabel("x:"))
        self.point_coords_layout.addWidget(self.x_edit)
        self.point_coords_layout.addWidget(QLabel("y:"))
        self.point_coords_layout.addWidget(self.y_edit)
        self.point_coords_layout.addWidget(QLabel("z:"))
        self.point_coords_layout.addWidget(self.z_edit)
        layout.addLayout(self.point_coords_layout)
        
        for label, coords in self.plane_coords.items():
            coord_layout = QHBoxLayout()
            coord_layout.addWidget(QLabel(f"{label} x:"))
            coord_layout.addWidget(coords["x"])
            coord_layout.addWidget(QLabel(f"{label} y:"))
            coord_layout.addWidget(coords["y"])
            coord_layout.addWidget(QLabel(f"{label} z:"))
            coord_layout.addWidget(coords["z"])
            layout.addLayout(coord_layout)
        
        layout.addWidget(self.add_button)
        self.setLayout(layout)
        self.on_object_type_changed("point")  # Set initial state
    
    def on_object_type_changed(self, current_text):
        # Show/hide input fields based on the selected object type
        if current_text == "point":
            for widget in [self.x_edit, self.y_edit, self.z_edit]:
                widget.show()
            for coords in self.plane_coords.values():
                for widget in coords.values():
                    widget.hide()
        else:
            for widget in [self.x_edit, self.y_edit, self.z_edit]:
                widget.hide()
            for coords in self.plane_coords.values():
                for widget in coords.values():
                    widget.show()
    
    def add_object(self):
        data = {
            "name": self.name_edit.text(),
            "type": self.object_type_combo.currentText()}
        if data['type'] == "point":
            data["x"] = self.x_edit.text()
            data["y"] = self.y_edit.text()
            data["z"] = self.z_edit.text()
        else:
            data['x'] = [self.plane_coords["lower left"]["x"].text(),
                    self.plane_coords["lower right"]["x"].text(),
                    self.plane_coords["upper left"]["x"].text()]


            data['y'] = [self.plane_coords["lower left"]["y"].text(),
                    self.plane_coords["lower right"]["y"].text(),
                    self.plane_coords["upper left"]["y"].text()]


            data['z'] = [self.plane_coords["lower left"]["z"].text(),
                    self.plane_coords["lower right"]["z"].text(),
                    self.plane_coords["upper left"]["z"].text()]

        
        # Emit the signal with the data
        self.object_added.emit(data)
        self.close()
