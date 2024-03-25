from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QWidget, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5 import QtCore, QtWidgets
import numpy as np
from numpy import cos
from mayavi.mlab import contour3d
import os
import segyio

os.environ['ETS_TOOLKIT'] = 'qt4'
from pyface.qt import QtGui as QtPyQt
from traits.api import HasTraits, Instance, on_trait_change, Property, Array
from traitsui.api import View, Item, HSplit, VSplit, InstanceEditor
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
from mayavi.core.ui.engine_view import EngineView
from mayavi import mlab


data_cube = None
######################################################################
class Mayavi(HasTraits):

    # The scene model.
    scene = Instance(MlabSceneModel, ())

    # The mayavi engine view.
    engine_view = Instance(EngineView)

    # The current selection in the engine tree view.
    current_selection = Property


    ######################
    view = View(HSplit(VSplit(Item(name='engine_view',
                                   style='custom',
                                   resizable=True,
                                   show_label=False
                                   ),
                              Item(name='current_selection',
                                   editor=InstanceEditor(),
                                   enabled_when='current_selection is not None',
                                   style='custom',
                                   springy=True,
                                   show_label=False),
                                   ),
                               Item(name='scene',
                                    editor=SceneEditor(),
                                    show_label=False,
                                    resizable=True,
                                    height=500,
                                    width=500),
                        ),
                resizable=True,
                scrollable=True
                )

    def __init__(self, **traits):
        HasTraits.__init__(self, **traits)
        self.engine_view = EngineView(engine=self.scene.engine)

        # Hook up the current_selection to change when the one in the engine
        # changes.  This is probably unnecessary in Traits3 since you can show
        # the UI of a sub-object in T3.
        self.scene.engine.on_trait_change(self._selection_change,
                                          'current_selection')
            
        
    def _selection_change(self, old, new):
        self.trait_property_changed('current_selection', old, new)

    def _get_current_selection(self):
        return self.scene.engine.current_selection
    
class MayaviQWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)  # Use QHBoxLayout for horizontal layout
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.visualization_maya = Mayavi()
    
        self.ui_maya = self.visualization_maya.edit_traits(parent=self,
                                                  kind='subpanel').control
        layout.addWidget(self.ui_maya)
        
class Ui_MainWindow(object):
    def __init__(self):
        super().__init__()

    @on_trait_change('scene3d.activated')
    def update_plot(self):
        global data_cube
        # This function is called when the view is opened. We don't
        # populate the scene when the view is not yet open, as some
        # VTK features require a GLContext.
        if data_cube is not None:
            s = mlab.pipeline.scalar_field(data_cube)
            s.spacing = [1, 1, -1]
            for axis in ['x', 'y', 'z']:
                plane = mlab.pipeline.image_plane_widget(s,
                                                        plane_orientation='{}_axes'.format(axis),
                                                        slice_index=100, colormap='gray')
                # Flip colormap. Better way to do this?
                plane.module_manager.scalar_lut_manager.reverse_lut = True

                # mlab.outline()   <------------with or without grid!!!
                mlab.axes(xlabel='Inline', ylabel='Crossline', zlabel='Depth', nb_labels=10)

    def load_seismic(self):
        global data_cube
        file_dialog = QFileDialog()
        filename, _ = file_dialog.getOpenFileName(self.centralwidget, 'Load SEG-Y File')
        if filename:
            with segyio.open(filename, iline=segyio.TraceField.INLINE_3D, xline=segyio.TraceField.CROSSLINE_3D) as stack:
                data_cube = segyio.cube(stack)
                data_cube = data_cube[:,:,950:1500]
        if filename:
            self.label.setText(f'SEG-Y File Selected: {filename}')
            self.update_plot()


    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setGeometry(200, 200, 800, 500)  # Adjusted width for both widgets

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)

        self.horizontalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setObjectName("menubar")
        self.horizontalLayout.addWidget(self.menubar)

        self.menuFiles = QtWidgets.QMenu(self.menubar)
        self.menuFiles.setObjectName("menuFiles")
        self.menubar.addMenu(self.menuFiles)

        self.actionLoad_segy = QtWidgets.QAction(MainWindow)
        self.actionLoad_segy.setObjectName("actionLoad_segy")
        self.menuFiles.addAction(self.actionLoad_segy)

        self.actionClose = QtWidgets.QAction(MainWindow)
        self.actionClose.setObjectName("actionClose")
        self.menuFiles.addAction(self.actionClose)

        self.plotArea = QWidget(self.centralwidget)
        self.plotArea.setObjectName("plotArea")
        self.plotLayout = QHBoxLayout(self.plotArea)
        self.plotLayout.setContentsMargins(0, 0, 0, 0)
        self.plotLayout.setSpacing(0)
        self.horizontalLayout.addWidget(self.plotArea)

        mayavi_widget = MayaviQWidget()
        self.plotLayout.addWidget(mayavi_widget)
        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.label = QLabel('No file selected')
        self.horizontalLayout.addWidget(self.label)
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        # Connect actions to Funcs methods
        self.actionLoad_segy.triggered.connect(self.load_seismic)
        self.actionClose.triggered.connect(self.close_application)

    def close_application(self):
        QApplication.quit()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Simulator"))
        self.menuFiles.setTitle(_translate("MainWindow", "Files"))
        self.actionLoad_segy.setText(_translate("MainWindow", "Load SEGY"))
        self.actionClose.setText(_translate("MainWindow", "Close"))

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())