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
import numpy as np
from PyQt5 import QtWidgets
from vispy.color import Colormap

from vispy.scene import SceneCanvas, visuals
from vispy.app import use_app
from vispy import scene, app
from vispy.visuals.transforms import STTransform, MatrixTransform

IMAGE_SHAPE = (600, 800)  # (height, width)
CANVAS_SIZE = (800, 600)  # (width, height)

import numpy as np
from vispy.scene import SceneCanvas, visuals
from vispy.app import use_app
from PyQt5 import QtWidgets, QtCore
from vispy.color import Colormap

class CanvasWrapper:
    def __init__(self):
        self.canvas = SceneCanvas(size=CANVAS_SIZE, show=True, keys='interactive')
        self.grid = self.canvas.central_widget.add_grid()
        
        # Load the seismic data
        self.vol = np.load('C:/Users/jp_reis/Downloads/train_seismic.npy')

        # Calculate percentiles
        self.percentiles = np.percentile(self.vol, [5, 25, 50, 75, 95])

        # Normalize the percentiles to the range [0, 1]
        self.normalized_percentiles = (self.percentiles - self.percentiles[0]) / (self.percentiles[-1] - self.percentiles[0])

        # Define the colors corresponding to the percentiles
        self.colors = [
            (0, 0, 1),    # Blue for the lowest percentile
            (0, 1, 1),    # Cyan
            (0, 1, 0),    # Green
            (1, 1, 0),    # Yellow
            (1, 0, 0)     # Red for the highest percentile
        ]
        
        # Create the custom colormap
        self.custom_colormap = Colormap(self.colors, controls=self.normalized_percentiles)

        # Initialize slice indices
        self.slice_x = self.vol.shape[0] // 2
        
        # Add views for each orthogonal slice
        self.view_x = self.grid.add_view(0, 0, bgcolor='white')
        
        # Create the images for each slice
        self.image_x = visuals.Image(self.vol[self.slice_x, :, ::-1].T, cmap=self.custom_colormap, parent=self.view_x.scene, interpolation = 'spline36')
        
        # Set up cameras
        self.view_x.camera = scene.PanZoomCamera(interactive=True)
        
        # Save the initial camera state (transform)
        self.initial_camera_transform = self.view_x.camera.get_state()
        self.view_x.camera.rect = (0, 0, self.vol.shape[1], self.vol.shape[2])
        self.initial_camera_rect = self.view_x.camera.rect
        
        # Connect key press event
        self.canvas.events.key_press.connect(self.on_key_press)
        
        self.update_view_ranges()

    def update_view_ranges(self):
        self.view_x.camera.set_range(x=(0, self.vol.shape[1]), y=(0, self.vol.shape[2]), margin=0)
        
    def update_slices(self, x):
        self.slice_x = x
        self.image_x.set_data(self.vol[self.slice_x, :, ::-1].T)
        self.canvas.update()

    def update_colormap(self, colormap):
        self.image_x.cmap = colormap

    def on_key_press(self, event):
        if event.text == ' ':
            self.view_x.camera.set_state(self.initial_camera_transform)
            self.view_x.camera.rect = self.initial_camera_rect
            self.view_x.camera.view_changed()

        
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

        # Slider
        self.slice_x_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slice_x_slider.setMinimum(0)
        self.slice_x_slider.setMaximum(self._canvas_wrapper.vol.shape[0] - 1)
        self.slice_x_slider.setValue(self._canvas_wrapper.slice_x)
        self.slice_x_slider.valueChanged.connect(self.on_slice_x_change)
        layout.addWidget(self.slice_x_slider)

        # Buttons for forward and backward
        self.backward_button = QtWidgets.QPushButton("Backward")
        self.backward_button.clicked.connect(self.on_backward_button_click)
        layout.addWidget(self.backward_button)

        self.forward_button = QtWidgets.QPushButton("Forward")
        self.forward_button.clicked.connect(self.on_forward_button_click)
        layout.addWidget(self.forward_button)

        # Colormap chooser
        self.colormap_label = QtWidgets.QLabel("Colormap:")
        layout.addWidget(self.colormap_label)
        self.colormap_chooser = QtWidgets.QComboBox()
        self.colormap_chooser.addItems(["gray", "seismic"])
        self.colormap_chooser.currentIndexChanged.connect(self.on_colormap_change)
        layout.addWidget(self.colormap_chooser)

        layout.addStretch(1)
        self.setLayout(layout)

    def on_slice_x_change(self, value):
        self._canvas_wrapper.update_slices(value)

    def on_backward_button_click(self):
        if self._canvas_wrapper.slice_x > 0:
            self._canvas_wrapper.slice_x -= 1
            self.slice_x_slider.setValue(self._canvas_wrapper.slice_x)
            self._canvas_wrapper.update_slices(self._canvas_wrapper.slice_x)

    def on_forward_button_click(self):
        if self._canvas_wrapper.slice_x < self._canvas_wrapper.vol.shape[0] - 1:
            self._canvas_wrapper.slice_x += 1
            self.slice_x_slider.setValue(self._canvas_wrapper.slice_x)
            self._canvas_wrapper.update_slices(self._canvas_wrapper.slice_x)
            
    def on_colormap_change(self, index):
        colormap = self.sender().currentText()
        self._canvas_wrapper.update_colormap(colormap)

if __name__ == "__main__":
    app = use_app("pyqt5")
    app.create()
    win = MyMainWindow()
    win.show()
    app.run()