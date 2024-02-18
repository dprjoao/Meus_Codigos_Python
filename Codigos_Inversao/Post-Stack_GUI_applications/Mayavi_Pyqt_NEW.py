import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog, QDialog
from mayavi import mlab
import numpy as np
import segyio


t = None
il = None
xl = None
t = None
data = None
data_cube = None

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Create widgets
        self.label = QLabel('No file selected')

        # Create button
        self.segy_button = QPushButton('Load SEG-Y File')

        # Connect button click event to function
        self.segy_button.clicked.connect(self.load_seismic)

        # Create layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.segy_button)

        # Set layout
        self.setLayout(layout)

        # Set window title and size
        self.setWindowTitle('File Loader')
        self.setGeometry(100, 100, 300, 200)

    def load_seismic(self):
        global data, data_cube
        file_dialog = QFileDialog()
        filename, _ = file_dialog.getOpenFileName(self, 'Load SEG-Y File')
        if filename:
            with segyio.open(filename, iline=segyio.TraceField.INLINE_3D, xline=segyio.TraceField.CROSSLINE_3D) as stack:
                ils, xls, twt = stack.ilines, stack.xlines, stack.samples
                data_cube = segyio.cube(stack)
                data_cube = data_cube[:,:,950:1500]
        if filename:
            #data = np.load(filename)
            self.label.setText(f'SEG-Y File Selected: {filename}')
            dialog = SuccessDialog()
            dialog.show_mayavi_scene()
            
class SuccessDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Success')
        self.setModal(True)
        self.setGeometry(300, 300, 800, 600)  # Set the size of the window

    def show_mayavi_scene(self):
        source = mlab.pipeline.scalar_field(data_cube)
        source.spacing = [1, 1, -1]

        for axis in ['x', 'y', 'z']:
            plane = mlab.pipeline.image_plane_widget(source, 
                                            plane_orientation='{}_axes'.format(axis),
                                            slice_index=100, colormap='gray')
            # Flip colormap. Better way to do this?
            plane.module_manager.scalar_lut_manager.reverse_lut = True
        mlab.axes(xlabel='Inline', ylabel='Crossline', zlabel='Depth', nb_labels=10) 
        mlab.outline()   
        mlab.show()

        
        mlab.test_plot3d()
        mlab.show()

if __name__ == '__main__':
    # Create QApplication instance
    app = QApplication(sys.argv)

    # Create instance of MyApp
    ex = MyApp()

    # Show the application
    ex.show()

    # Run the application loop
    sys.exit(app.exec_())
