import seismic_canvas
import numpy as np
from PyQt5 import QtWidgets, QtCore
from vispy.scene import SceneCanvas, visuals
from vispy.app import use_app
from vispy import scene
import h5py

IMAGE_SHAPE = (400, 300)  # (height, width)
CANVAS_SIZE = (900, 900)  # (width, height)

class CanvasWrapper:
    def __init__(self, size=CANVAS_SIZE, keys='interactive'):
        self.size = size
        self.keys = keys
        self.canvas = SceneCanvas(size=self.size, show=True, keys=self.keys, resizable=True)
        self.grid = self.canvas.central_widget.add_grid()
        
        # Initialize the HDF5 dataset reference
        self.load_data()

        # Initialize slice indices
        self.slice_il = self.dataset.shape[0] // 2
        
        # Add views for each orthogonal slice
        self.view_il = self.grid.add_view(0, 0, bgcolor='white')

        data = self.dataset[self.slice_il,:,:].T
        
        # Create the images for each slice
        self.image_il = visuals.Image(data[::-1,:], cmap="gray", parent=self.view_il.scene)
        
        # Set up cameras
        self.view_il.camera = scene.PanZoomCamera(interactive=True)
        self.initial_camera_transform = self.view_il.camera.get_state()
        self.view_il.camera.rect = (0, 0, self.dataset.shape[1], self.dataset.shape[2])
        self.initial_camera_rect = self.view_il.camera.rect

        # Connect key press event
        self.canvas.events.key_press.connect(self.on_key_press)
        self.update_view_ranges()

    def load_data(self, filepath='seismic_data_chunked.h5', dtype=np.float32):
        # Open the HDF5 file and get a reference to the dataset
        self.hdf5_file = h5py.File(filepath, 'r')
        self.dataset = self.hdf5_file['data']

    def update_view_ranges(self):
        self.view_il.camera.set_range(x=(0, self.dataset.shape[1]), y=(0, self.dataset.shape[2]), margin=0)

    def update_slices(self, x):
        # Load only the required slice from the dataset
        self.slice_x = x
        img = self.dataset[self.slice_x, :, :].T
        self.image_il.set_data(img[::-1, :])
        self.canvas.update()

    def update_colormap(self, colormap):
        self.image_il.cmap = colormap

    def on_key_press(self, event):
        if event.text == ' ':
            self.view_il.camera.set_state(self.initial_camera_transform)
            self.view_il.camera.rect = self.initial_camera_rect
            self.view_il.camera.view_changed()

    def __del__(self):
        # Close the HDF5 file when the object is deleted
        if hasattr(self, 'hdf5_file'):
            self.hdf5_file.close()


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
        self.slice_il_label = QtWidgets.QLabel("Inline Slice Index:")
        layout.addWidget(self.slice_il_label)

        self.slice_il_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slice_il_slider.setMinimum(0)
        self.slice_il_slider.setMaximum(self._canvas_wrapper.dataset.shape[0] - 1)
        self.slice_il_slider.setValue(self._canvas_wrapper.slice_il)
        self.slice_il_slider.valueChanged.connect(self.on_slice_il_change)
        layout.addWidget(self.slice_il_slider)

        self.backward_button = QtWidgets.QPushButton("Backward")
        self.backward_button.clicked.connect(self.on_backward_button_click)
        layout.addWidget(self.backward_button)

        self.forward_button = QtWidgets.QPushButton("Forward")
        self.forward_button.clicked.connect(self.on_forward_button_click)
        layout.addWidget(self.forward_button)

        self.colormap_label = QtWidgets.QLabel("Colormap:")
        layout.addWidget(self.colormap_label)
        self.colormap_chooser = QtWidgets.QComboBox()
        self.colormap_chooser.addItems(["gray", "seismic"])
        self.colormap_chooser.currentIndexChanged.connect(self.on_colormap_change)
        layout.addWidget(self.colormap_chooser)

        layout.addStretch(1)
        self.setLayout(layout)

    def on_slice_il_change(self, value):
        self._canvas_wrapper.update_slices(value)

    def on_backward_button_click(self):
        if self._canvas_wrapper.slice_il > 0:
            self._canvas_wrapper.slice_il -= 1
            self.slice_il_slider.setValue(self._canvas_wrapper.slice_il)
            self._canvas_wrapper.update_slices(self._canvas_wrapper.slice_il)

    def on_forward_button_click(self):
        if self._canvas_wrapper.slice_il < self._canvas_wrapper.vol.shape[0] - 1:
            self._canvas_wrapper.slice_il += 1
            self.slice_il_slider.setValue(self._canvas_wrapper.slice_il)
            self._canvas_wrapper.update_slices(self._canvas_wrapper.slice_il)
            
    def on_colormap_change(self, index):
        colormap = self.sender().currentText()
        self._canvas_wrapper.update_colormap(colormap)

if __name__ == "__main__":
    app = use_app("pyqt5")
    app.create()
    win = MyMainWindow()
    win.show()
    app.run()
