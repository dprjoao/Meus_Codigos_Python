import tkinter as tk
from tkinter import Toplevel, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import RangeSlider
import numpy as np
import segyio
import matplotlib
matplotlib.use('TkAgg')

class MyApp():
    #def __init__(self, root):
        #self.root = root
        #self.root.title("Matplotlib in Tkinter")
        
        # Create buttons
        #self.button_load = tk.Button(self.root, text="Load File", command=self.load_file)
        #self.button_load.pack()

    def load_file(self):
        global il, xl, t, data_cube
        filename = filedialog.askopenfilename()
        if filename:
            with segyio.open(filename, iline=189, xline=193) as stack:
                ils, xls, twt = stack.ilines, stack.xlines, stack.samples
                data_cube = segyio.cube(stack)
                data_cube[np.isnan(data_cube)] = 0.0
                n_traces = stack.tracecount 
                tr = stack.attributes(segyio.TraceField.TraceNumber)[-1]
                if not isinstance(tr, int):
                    tr = stack.attributes(segyio.TraceField.TraceNumber)[-2] + 1
                tr = int(tr[0])

            il, xl, t = ils, xls, twt
            il_start, il_end = il[0], il[-1]
            xl_start, xl_end = xl[0], xl[-1]
            dt = t[1] - t[0]

            fig_slc = plt.figure(figsize=(10, 6))
            gs = GridSpec(2, 2, height_ratios=(10,1))

            ax_seismic = fig_slc.add_subplot(gs[:1,:])
            ax_histogram = fig_slc.add_subplot(gs[-1,0])         
            plt.subplots_adjust(left=0.098, right=1, top=0.955, bottom=0.112,hspace=0.25)
            im = ax_seismic.imshow(data_cube.T, aspect='auto', 
                                    cmap='gray_r', vmin=0.1*data_cube.min(), 
                                    vmax=0.1*data_cube.max(),
                                    extent=[xl_start, xl_end, t[-1], t[0]])
            
            ax_histogram.hist(data_cube.T.flatten(), bins=400)
            ax_histogram.set_title('Histogram of pixel intensities')
            fig_slc.colorbar(im, ax=ax_seismic)
            
            axframe1 = plt.axes([0.1, 0.01, 0.5, 0.03], facecolor='lightgoldenrodyellow')
            slider = RangeSlider(axframe1, "Threshold",
                                -0.1*data_cube.max(), 
                                0.1*data_cube.max())

            lower_limit_line = ax_histogram.axvline(slider.val[0], color='k')
            upper_limit_line = ax_histogram.axvline(slider.val[1], color='k')

            plt.grid(False)
            plot_window = Toplevel(self.root)
            plot_window.title("Data Cube Plot")
            canvas = FigureCanvasTkAgg(fig_slc, master=plot_window)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True)
            
            # Add navigation toolbar
            toolbar = NavigationToolbar2Tk(canvas, plot_window)

            def update_slc(val):
                im.norm.vmin = val[0]
                im.norm.vmax = val[1]
                lower_limit_line.set_xdata([val[0], val[0]])
                upper_limit_line.set_xdata([val[1], val[1]])
                fig_slc.canvas.draw()
                #canvas.draw_idle()  # Redraw canvas

            
            slider.on_changed(update_slc)
            canvas.draw()
            # Connect the slider to the update_slc function
            

class App(MyApp):
    # Python constructor and Tkinter mainloop
    def __init__(self) -> None:
        self.root = tk.Tk()
        #app = MyApp(self.root)
        self.root.title("Matplotlib in Tkinter")
        self.widgets()
        self.root.mainloop()
        
        
    def widgets(self):
        # Create buttons
        self.button_load = tk.Button(self.root, text="Load File", command=self.load_file)
        self.button_load.pack()


App()
#if __name__ == "__main__":
#    main()