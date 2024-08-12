# -*- coding: utf-8 -*-
# vispy: gallery 2
# -----------------------------------------------------------------------------
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
"""
Embed VisPy into Qt
===================

Display VisPy visualizations in a PyQt5 application.

"""
import seismic_canvas
import numpy as np
from PyQt5 import QtWidgets

from vispy.scene import SceneCanvas, visuals
from vispy.app import use_app
from vispy import scene
from vispy.visuals.transforms import STTransform

IMAGE_SHAPE = (600, 800)  # (height, width)
CANVAS_SIZE = (800, 600)  # (width, height)
NUM_LINE_POINTS = 200

import numpy as np
from vispy.scene import SceneCanvas, visuals
from vispy.app import use_app
from PyQt5 import QtWidgets, QtCore

class CanvasWrapper:
    def __init__(self):
        self.canvas = SceneCanvas(size=CANVAS_SIZE, show=True, keys='interactive')
        self.grid = self.canvas.central_widget.add_grid()
        
        # Load the seismic data
        self.vol = np.load('C:/Users/jp_reis/Downloads/train_seismic.npy')

        # Initialize slice indices
        self.slice_x = self.vol.shape[0] // 2
        self.slice_y = self.vol.shape[1] // 2
        self.slice_z = self.vol.shape[2] // 2
        
        # Add views for each orthogonal slice
        self.view_x = self.grid.add_view(0, 0, bgcolor='white')
        self.view_y = self.grid.add_view(0, 1, bgcolor='white')
        self.view_z = self.grid.add_view(1, 0, bgcolor='white')

        # Create the images for each slice
        self.image_x = visuals.Image(self.vol[self.slice_x, :, ::-1].T, cmap="viridis", parent=self.view_x.scene)
        self.image_y = visuals.Image(self.vol[:, self.slice_y, ::-1].T, cmap="viridis", parent=self.view_y.scene)
        self.image_z = visuals.Image(self.vol[:, :, self.slice_z], cmap="viridis", parent=self.view_z.scene)
        
        # Set up cameras
        self.view_x.camera = 'panzoom'
        self.view_y.camera = 'panzoom'
        self.view_z.camera = 'panzoom'

        self.update_view_ranges()

    def update_view_ranges(self):
        # Set the range for each view
        self.view_x.camera.set_range(x=(0, self.vol.shape[1]), y=(0, self.vol.shape[2]), margin=0)
        self.view_y.camera.set_range(x=(0, self.vol.shape[0]), y=(0, self.vol.shape[2]), margin=0)
        self.view_z.camera.set_range(x=(0, self.vol.shape[1]), y=(0, self.vol.shape[0]), margin=0)

    def update_slices(self, x, y, z):
        self.slice_x = x
        self.slice_y = y
        self.slice_z = z

        self.image_x.set_data(self.vol[self.slice_x, :, ::-1].T)
        self.image_y.set_data(self.vol[:, self.slice_y, ::-1].T)
        self.image_z.set_data(self.vol[:, :, self.slice_z])

        self.canvas.update()

    def update_colormap(self, colormap):
        self.image_x.cmap = colormap
        self.image_y.cmap = colormap
        self.image_z.cmap = colormap

class MyMainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout()

        self._canvas_wrapper = CanvasWrapper()
        self._controls = Controls(self._canvas_wrapper)
        main_layout.addWidget(self._controls)
        main_layout.addWidget(self._canvas_wrapper.canvas.native)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

class Controls(QtWidgets.QWidget):
    def __init__(self, canvas_wrapper, parent=None):
        super().__init__(parent)
        self._canvas_wrapper = canvas_wrapper

        layout = QtWidgets.QVBoxLayout()
        
        # Slice sliders
        self.slice_x_label = QtWidgets.QLabel("X Slice Index:")
        layout.addWidget(self.slice_x_label)
        self.slice_x_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slice_x_slider.setMinimum(0)
        self.slice_x_slider.setMaximum(self._canvas_wrapper.vol.shape[0] - 1)
        self.slice_x_slider.setValue(self._canvas_wrapper.slice_x)
        self.slice_x_slider.valueChanged.connect(self.on_slice_x_change)
        layout.addWidget(self.slice_x_slider)

        self.slice_y_label = QtWidgets.QLabel("Y Slice Index:")
        layout.addWidget(self.slice_y_label)
        self.slice_y_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slice_y_slider.setMinimum(0)
        self.slice_y_slider.setMaximum(self._canvas_wrapper.vol.shape[1] - 1)
        self.slice_y_slider.setValue(self._canvas_wrapper.slice_y)
        self.slice_y_slider.valueChanged.connect(self.on_slice_y_change)
        layout.addWidget(self.slice_y_slider)

        self.slice_z_label = QtWidgets.QLabel("Z Slice Index:")
        layout.addWidget(self.slice_z_label)
        self.slice_z_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slice_z_slider.setMinimum(0)
        self.slice_z_slider.setMaximum(self._canvas_wrapper.vol.shape[2] - 1)
        self.slice_z_slider.setValue(self._canvas_wrapper.slice_z)
        self.slice_z_slider.valueChanged.connect(self.on_slice_z_change)
        layout.addWidget(self.slice_z_slider)

        # Colormap chooser
        self.colormap_label = QtWidgets.QLabel("Colormap:")
        layout.addWidget(self.colormap_label)
        self.colormap_chooser = QtWidgets.QComboBox()
        self.colormap_chooser.addItems(["viridis", "reds", "blues"])
        self.colormap_chooser.currentIndexChanged.connect(self.on_colormap_change)
        layout.addWidget(self.colormap_chooser)

        layout.addStretch(1)
        self.setLayout(layout)

    def on_slice_x_change(self, value):
        self._canvas_wrapper.update_slices(value, self.slice_y_slider.value(), self.slice_z_slider.value())

    def on_slice_y_change(self, value):
        self._canvas_wrapper.update_slices(self.slice_x_slider.value(), value, self.slice_z_slider.value())

    def on_slice_z_change(self, value):
        self._canvas_wrapper.update_slices(self.slice_x_slider.value(), self.slice_y_slider.value(), value)

    def on_colormap_change(self, index):
        colormap = self.sender().currentText()
        self._canvas_wrapper.update_colormap(colormap)

if __name__ == "__main__":
    app = use_app("pyqt5")
    app.create()
    win = MyMainWindow()
    win.show()
    app.run()