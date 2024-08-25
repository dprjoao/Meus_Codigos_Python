import sys
import numpy as np
from vispy import app, scene
from vispy.color import Colormap
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget
import matplotlib.pyplot as plt

class VisualizationWidget(QWidget):
    def __init__(self, seismic_data):
        super().__init__()
        self.data = seismic_data
        self.canvas = scene.SceneCanvas(keys='interactive', show=True)
        self.view = self.canvas.central_widget.add_view()  # Correct way to add a view
        
        # Initialize with the first slice
        self.current_slice = 0
        self.image = scene.visuals.Image(self.data[:, :, self.current_slice], cmap='viridis', clim=(self.percentile(5), self.percentile(95)))
        self.view.add(self.image)
        self.view.camera = 'panzoom'

        # Create layout and add canvas
        layout = QVBoxLayout()
        layout.addWidget(self.canvas.native)
        
        # Button to update color limits
        self.button = QPushButton("Update Color Limits")
        self.button.clicked.connect(self.update_color_limits)
        layout.addWidget(self.button)
        
        # Button to cycle through slices
        self.slice_button = QPushButton("Next Slice")
        self.slice_button.clicked.connect(self.next_slice)
        layout.addWidget(self.slice_button)

        # Set layout
        self.setLayout(layout)
        
        # Initial histogram plotting
        self.plot_histogram()

    def percentile(self, p):
        return np.percentile(self.data, p)

    def set_clim(self, pmin, pmax):
        min_value = self.percentile(pmin)
        max_value = self.percentile(pmax)
        self.image.clim = (min_value, max_value)
        self.canvas.update()

    def update_color_limits(self):
        # Example: dynamically adjust color limits
        self.set_clim(10, 90)

    def next_slice(self):
        self.current_slice = (self.current_slice + 1) % self.data.shape[2]
        self.image.set_data(self.data[:, :, self.current_slice])
        self.canvas.update()

    def plot_histogram(self):
        # Plot histogram of seismic data
        amplitudes = self.data.flatten()
        plt.figure()
        plt.hist(amplitudes, bins=50, edgecolor='k', alpha=0.7)
        plt.axvline(self.percentile(5), color='r', linestyle='dashed', linewidth=1, label='5th Percentile')
        plt.axvline(self.percentile(95), color='g', linestyle='dashed', linewidth=1, label='95th Percentile')
        plt.xlabel('Amplitude')
        plt.ylabel('Frequency')
        plt.title('Histogram of Seismic Data Amplitudes')
        plt.legend()
        plt.show()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Seismic Data Visualization")
        self.setGeometry(100, 100, 800, 600)
        
        # Sample seismic data (replace with your actual data)
        seismic_data = np.random.randn(100, 100, 100)
        
        # Create visualization widget
        self.widget = VisualizationWidget(seismic_data)
        self.setCentralWidget(self.widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
