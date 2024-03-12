from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, filedialog, Label, Frame, Toplevel, IntVar, Radiobutton, ttk
import segyio
import numpy as np
import matplotlib.pyplot as plt
import pylops
import os
import sys
import matplotlib
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import RangeSlider
import pandas as pd
matplotlib.use('TkAgg')

#Global variables
t = None
il = None
xl = None
t = None
dt = None
wav_est = None
wavs = None
m_tbt = None
tmin = None
tmax = None 
epsI_entry = None

# Get the path of the executable
executable_path = sys.argv[0]

# Navigate one directory up
parent_directory = os.path.dirname(executable_path)

#Join with "assets" folder
ASSETS_PATH = os.path.join(parent_directory, "assets/frame0")

#Python object with computing methods
class Funcs():    
    #Data Loading
    def load_file(self):
        global t, data_cube, il, xl, dt
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
            self.status_label.config(text=f"File loaded: {filename}")
            # Enable the button to open the parameter input window
            self.button_load.config(state="normal")
            print("File loaded successfully.")
            # Define data as 2D/3D and Post-stack/Pre-stack
            if len(data_cube.shape) == 3:
                if data_cube.shape[0] != 1:
                    data_type = 'Post-stack 3D'
                else:
                    if n_traces > tr > 1:   
                        data_type = 'Post-stack 3D'
                    else:
                        data_type = 'Post-stack 2D'
                if data_type == 'Post-stack 3D':
                    fig_slc = plt.figure(figsize=(10, 6))
            
                    #fig.subplots_adjust(bottom=0.25)
                    gs = GridSpec(2, 2, height_ratios=(10,1))

                    ax_seismic = fig_slc.add_subplot(gs[:1,:])
                    ax_histogram = fig_slc.add_subplot(gs[-1,0])         
                    plt.subplots_adjust(left=0.098, right=1, top=0.955, bottom=0.112,hspace=0.25)
                    im = ax_seismic.imshow(data_cube[..., int(4000/dt)], aspect='auto', 
                                           cmap='gray_r', vmin = 0.1*data_cube[..., int(4000/dt)].min(), 
                                           vmax = 0.1*data_cube[..., int(4000/dt)].max(),
                                extent=[xl_start, xl_end, il_start, il_end])
                    
                    ax_histogram.hist(data_cube[..., int(4000/dt)].T.flatten(), bins = 400)
                    ax_histogram.set_title('Histogram of pixel intensities')
                    plt.colorbar(im, ax=ax_seismic)

                    # Create the RangeSlider
                    # Add slider for interactive frame navigation along inline direction
                    axframe1 = plt.axes([0.1, 0.01, 0.5, 0.03], facecolor='lightgoldenrodyellow')
                    #slider_ax = fig.add_axes([0.20, 0.1, 0.60, 0.03])
                    slider = RangeSlider(axframe1, "Threshold",
                                        -0.1*data_cube[..., int(4000/dt)].max(), 
                                         0.1*data_cube[..., int(4000/dt)].max())

                    # Create the Vertical lines on the histogram
                    lower_limit_line = ax_histogram.axvline(slider.val[0], color='k')
                    upper_limit_line = ax_histogram.axvline(slider.val[1], color='k')


                    def update_slc(val):
                        # The val passed to a callback by the RangeSlider will
                        # be a tuple of (min, max)

                        # Update the image's colormap
                        im.norm.vmin = val[0]
                        im.norm.vmax = val[1]

                        # Update the position of the vertical lines
                        lower_limit_line.set_xdata([val[0], val[0]])
                        upper_limit_line.set_xdata([val[1], val[1]])

                        # Redraw the figure to ensure it updates
                        #fig_slc.canvas.draw_idle()


                    slider.on_changed(update_slc)
                    plt.grid(False)
                    plt.show()
                    '''fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                    c = ax.imshow(data_cube[..., int(4000/dt)], aspect='auto', cmap='gray_r', vmin = 0.1*data_cube[..., int(4000/dt)].min(), vmax = 0.1*data_cube[..., int(4000/dt)].max(),
                                extent=[xl_start, xl_end, il_start, il_end])
                    plt.colorbar(c, ax=ax, pad=0.01)
                    plt.grid(False)
                    plt.show()'''

                else:
                    pass           
                return data_cube, il, xl, t, dt
            else:
                return None, None, None, None
                
    # Function to get IL number from input
    def get_iline_num(self):
        global il_number
        try:
            il_number = int(self.il_entry.get())
            self.il_label.config(text=f"Selected IL: {il_number}")

            il_start= il[0]
            xl_start, xl_end = xl[0], xl[-1]
            fig_il = plt.figure(figsize=(10, 6))
            
            #fig.subplots_adjust(bottom=0.25)
            gs = GridSpec(2, 2, height_ratios=(10,1))

            ax_seismic = fig_il.add_subplot(gs[:1,:])
            ax_histogram = fig_il.add_subplot(gs[-1,0])         
            plt.subplots_adjust(left=0.098, right=1, top=0.955, bottom=0.112,hspace=0.25)
            im = ax_seismic.imshow(data_cube[il_number - il_start, :, :].T, aspect='auto', cmap='gray_r', 
                                   vmin = 0.1*data_cube[il_number - il_start, :, :].min(), 
                                   vmax = 0.1*data_cube[il_number - il_start, :, :].max(),
                            extent=[xl_start, xl_end, t[-1], t[0]])
            
            ax_histogram.hist(data_cube[:,0:300,:].T.flatten(), bins = 400)
            ax_histogram.set_title('Histogram of pixel intensities')
            plt.colorbar(im, ax=ax_seismic)
            # Create the RangeSlider
            # Add slider for interactive frame navigation along inline direction
            axframe1 = plt.axes([0.1, 0.01, 0.5, 0.03], facecolor='lightgoldenrodyellow')
            #slider_ax = fig.add_axes([0.20, 0.1, 0.60, 0.03])
            slider = RangeSlider(axframe1, "Threshold", -0.1*data_cube[il_number - il_start, :, :].max(), 0.1*data_cube[il_number - il_start, :, :].max())

            # Create the Vertical lines on the histogram
            lower_limit_line = ax_histogram.axvline(slider.val[0], color='k')
            upper_limit_line = ax_histogram.axvline(slider.val[1], color='k')


            def update_il(val):
                # The val passed to a callback by the RangeSlider will
                # be a tuple of (min, max)

                # Update the image's colormap
                im.norm.vmin = val[0]
                im.norm.vmax = val[1]

                # Update the position of the vertical lines
                lower_limit_line.set_xdata([val[0], val[0]])
                upper_limit_line.set_xdata([val[1], val[1]])

                # Redraw the figure to ensure it updates
                #fig_il.canvas.draw_idle()


            slider.on_changed(update_il)
            plt.grid(False)
            plt.show()
            #fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            #c = ax.imshow(data_cube[il_number - il_start, :, :].T, aspect='auto', cmap='gray_r', vmin = 0.1*data_cube[il_number - il_start, :, :].min(), vmax = 0.1*data_cube[il_number - il_start, :, :].max(),
            #                extent=[xl_start, xl_end, t[-1], t[0]])
            #plt.colorbar(c, ax=ax, pad=0.01)
            #plt.grid(False)
            return il_number
        except ValueError:
            self.status_label.config(text="Error: Please enter valid IL number.")
            return None
    # Parameter window for wavelet
    def open_parameter_window(self):
        parameter_window = Toplevel(self.root)
        parameter_window.title("Parameter Input")

        # Create labels and entry widgets for parameter input
        tmin_label = Label(parameter_window, text="Enter time min:")
        tmin_label.grid(row=0, column=0)

        self.tmin_entry = Entry(parameter_window)
        self.tmin_entry.grid(row=0, column=1)

        tmax_label = Label(parameter_window, text="Enter max time:")
        tmax_label.grid(row=1, column=0)

        self.tmax_entry = Entry(parameter_window)
        self.tmax_entry.grid(row=1, column=1)
        if wavs is not None:
            estimate_button = Button(parameter_window, text="Estimate Wavelet", command=self.compute_mean_wvlt)
        else:
            estimate_button = Button(parameter_window, text="Estimate Wavelet", command=self.estimate_wavelet)

        estimate_button.grid(row=3,column=1)
            
        close_button = Button(parameter_window, text="Close", command= parameter_window.destroy)
        close_button.grid(row=4,column=1)
    # Method for inputing tied wavelet in .txt formatt
    def input_tied_wvlt(self):
        global wav_est, wavs, times
        
        # Open file dialog to select multiple files
        wav_files = filedialog.askopenfilenames()

        if wav_files:
            wavs = []
            times = []
            for file in wav_files:
                arr = pd.read_csv(file, skiprows=16, sep='\s+').to_numpy()
                wavs.append(arr[:, 1] / np.max(abs(arr[:, 1])))
                times.append(arr[:, 0])

            wavs = np.array(wavs)
            times = np.array(times)

            # Example of plotting
            import matplotlib.pyplot as plt
            plt.figure(figsize=(4, 3))
            for i in range(len(wavs)):
                plt.plot(times[i], wavs[i])
            plt.show() 

    def compute_mean_wvlt(self):
        global wav_est
        self.tmin = float(self.tmin_entry.get())
        self.tmax = float(self.tmax_entry.get())
        wav = np.mean(wavs, axis=0)
        wav = wav / np.max(abs(wav))
        wav_est = np.concatenate((wav[0:17], wav[0:14][::-1]), axis=0)

        freqs = np.fft.rfftfreq(times.shape[1]-1, d=dt/1000)
        a = np.fft.rfft(wav_est)
        A = np.abs(a)

        # display wavelet
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle('Statistical wavelet estimate')
        axs[0].plot(freqs, A, 'k')
        axs[0].set_title('Frequency')
        axs[1].plot( wav_est, 'k')
        axs[1].set_title('Time');
            
        plt.show()        
    # Method for wavelet estimation
    def estimate_wavelet(self):
        global data_cube, t, wav_est
        self.tmin = float(self.tmin_entry.get())
        self.tmax = float(self.tmax_entry.get())
        nt_wav = 16
        nfft = 2**8
        wav_est_fft = np.mean(np.abs(np.fft.fft(data_cube[..., int(self.tmin/dt):int(self.tmax/dt)], nfft, axis=-1)), axis=(0, 1))
        fwest = np.fft.fftfreq(nfft, d=dt/1000)
        wav_est = np.real(np.fft.ifft(wav_est_fft)[:nt_wav])
        wav_est = np.concatenate((np.flipud(wav_est[1:]), wav_est), axis=0)
        wav_est = wav_est / wav_est.max()
        
        '''fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        c = plt.imshow(data_cube[il_number - il[0], :, int(self.tmin/dt):int(self.tmax/dt)].T, aspect='auto', cmap='gray_r', vmin = 0.1*data_cube[il_number - il[0], :, int(self.tmin/dt):int(self.tmax/dt)].T.min(), vmax = 0.1*data_cube[il_number - il[0], :, int(self.tmin/dt):int(self.tmax/dt)].T .max(),
                            extent=[xl[0], xl[-1], t[int(self.tmax/dt)], t[int(self.tmin/dt)]])
        plt.colorbar(c, ax=ax, pad=0.01)
        ax.set_title('Interval section')
        plt.grid(False)'''

        fig_sect = plt.figure(figsize=(10, 5))
        gs = GridSpec(2, 2, height_ratios=(10,1))
        ax_seismic = fig_sect.add_subplot(gs[:1,:])
        ax_histogram = fig_sect.add_subplot(gs[-1,0])  
        plt.subplots_adjust(left=0.098, right=1, top=0.955, bottom=0.112,hspace=0.25)            
        im = ax_seismic.imshow(data_cube[il_number - il[0], :, int(self.tmin/dt):int(self.tmax/dt)].T,
                                aspect='auto', cmap='gray_r',
                                vmin = 0.1*data_cube[il_number - il[0], :, int(self.tmin/dt):int(self.tmax/dt)].T.min(),
                                vmax = 0.1*data_cube[il_number - il[0], :, int(self.tmin/dt):int(self.tmax/dt)].T.max(),
                            extent=[xl[0], xl[-1], t[int(self.tmax/dt)], t[int(self.tmin/dt)]])
        ax_histogram.hist(data_cube[il_number - il[0], :, int(self.tmin/dt):int(self.tmax/dt)].T.flatten(), bins = 400)
        ax_histogram.set_title('Histogram of pixel intensities')
        plt.colorbar(im, ax=ax_seismic)

        # Create the RangeSlider
        # Add slider for interactive frame navigation along inline direction
        axframe1 = plt.axes([0.1, 0.01, 0.5, 0.03], facecolor='lightgoldenrodyellow')
        #slider_ax = fig.add_axes([0.20, 0.1, 0.60, 0.03])
        slider = RangeSlider(axframe1, "Threshold", data_cube[il_number - il[0], :, int(self.tmin/dt):int(self.tmax/dt)].min(),
                            data_cube[il_number - il[0], :, int(self.tmin/dt):int(self.tmax/dt)].max())
        # Create the Vertical lines on the histogram
        lower_limit_line = ax_histogram.axvline(slider.val[0], color='k')
        upper_limit_line = ax_histogram.axvline(slider.val[1], color='k')

        #display wavelet
        fig2, axs = plt.subplots(1, 2, figsize=(10, 5))
        fig2.suptitle('Statistical wavelet estimate')
        axs[0].plot(fwest[:nfft//2], wav_est_fft[:nfft//2], 'k')
        axs[0].set_title('Frequency')
        axs[1].plot(wav_est, 'k')
        axs[1].set_title('Time')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.22)
        plt.grid(False)
        def update(val):
            # The val passed to a callback by the RangeSlider will
            # be a tuple of (min, max)

            # Update the image's colormap
            im.norm.vmin = val[0]
            im.norm.vmax = val[1]

            # Update the position of the vertical lines
            lower_limit_line.set_xdata([val[0], val[0]])
            upper_limit_line.set_xdata([val[1], val[1]])

            # Redraw the figure to ensure it updates
            #fig_sect.canvas.draw_idle()
        slider.on_changed(update)
        plt.show()
    # Trace-by-trace inversion parameter window
    def tbt_inv_param_win(self, mode):
        global epsI_entry
        inv_param_win = Toplevel(self.root)
        inv_param_win.title("Parameter Input")

        epsI_label = Label(inv_param_win, text="Damping (epsI):")
        epsI_label.grid(row=0, column=0)
        epsI_entry = Entry(inv_param_win)
        epsI_entry.grid(row=0, column=1)

        run_inv_button = Button(inv_param_win, text="Run inversion", command = lambda: self.run_inversion(mode))
        run_inv_button.grid(row=1,column=1)
            
        close_button = Button(inv_param_win, text="Close", command= inv_param_win.destroy)
        close_button.grid(row=2,column=1)
        return epsI_entry
    # Spatially regularized inversion parameter window
    def spat_inv_param_win(self, mode):
        global niter_sr_entry, epsI_sr_entry, epsR_sr_entry
        inv_param_win = Toplevel(self.root)
        inv_param_win.title("Parameter Input")

        # Create labels and entry widgets for parameter input
        niter_label = Label(inv_param_win, text="Number of iterations:")
        niter_label.grid(row=0, column=0)
        niter_sr_entry = Entry(inv_param_win)
        niter_sr_entry.grid(row=0, column=1)

        epsI_label = Label(inv_param_win, text="Damping (epsI):")
        epsI_label.grid(row=1, column=0)
        epsI_sr_entry = Entry(inv_param_win)
        epsI_sr_entry.grid(row=1, column=1)

        epsR_label = Label(inv_param_win, text="Spatial regularization (epsR):")
        epsR_label.grid(row=2, column=0)
        epsR_sr_entry = Entry(inv_param_win)
        epsR_sr_entry.grid(row=2, column=1)

        run_inv_button = Button(inv_param_win, text="Run inversion", command = lambda: self.run_inversion(mode))
        run_inv_button.grid(row=3,column=1)
        
        close_button = Button(inv_param_win, text="Close", command= inv_param_win.destroy)
        close_button.grid(row=4,column=1)
        return niter_sr_entry, epsI_sr_entry, epsR_sr_entry
    # Blocky reg inv parameter window
    def blocky_inv_param_win(self, mode):
        global niter_out_b_entry, niter_in_b_entry, mu_b_entry, epsI_b_entry, epsR_b_entry, epsRL1_b_entry
        inv_param_win = Toplevel(self.root)
        inv_param_win.title("Parameter Input")
        
        # Create labels and entry widgets for parameter input
        niter_out_label = Label(inv_param_win, text="Number of outer loop iterations:")
        niter_out_label.grid(row=0, column=0)
        niter_out_b_entry = Entry(inv_param_win)
        niter_out_b_entry.grid(row=0, column=1)
        niter_in_b_label = Label(inv_param_win, text="Number of inner loop iterations:")
        niter_in_b_label.grid(row=1, column=0)
        niter_in_b_entry = Entry(inv_param_win)
        niter_in_b_entry.grid(row=1, column=1)

        niter_b_label = Label(inv_param_win, text="Number of iterations of lsqr:")
        niter_b_label.grid(row=2, column=0)
        niter_b_entry = Entry(inv_param_win)
        niter_b_entry.grid(row=2, column=1)

        mu_b_label = Label(inv_param_win, text="Data term damping:")
        mu_b_label.grid(row=3, column=0)
        mu_b_entry = Entry(inv_param_win)
        mu_b_entry.grid(row=3, column=1)

        epsI_b_label = Label(inv_param_win, text="Damping:")
        epsI_b_label.grid(row=4, column=0)
        epsI_b_entry = Entry(inv_param_win)
        epsI_b_entry.grid(row=4, column=1)

        epsR_b_label = Label(inv_param_win, text="Spatial regularization:")
        epsR_b_label.grid(row=5, column=0)
        epsR_b_entry = Entry(inv_param_win)
        epsR_b_entry.grid(row=5, column=1)

        epsRL1_b_label = Label(inv_param_win, text="Blocky regularization:")
        epsRL1_b_label.grid(row=6, column=0)
        epsRL1_b_entry = Entry(inv_param_win)
        epsRL1_b_entry.grid(row=6, column=1)

        run_inv_button = Button(inv_param_win, text="Run inversion", command = lambda: self.run_inversion(mode))
        run_inv_button.grid(row=7,column=1)
            
        close_button = Button(inv_param_win, text="Close", command= inv_param_win.destroy)
        close_button.grid(row=8,column=1)
        return niter_out_b_entry, niter_in_b_entry, mu_b_entry, epsI_b_entry, epsR_b_entry, epsRL1_b_entry
    
    # Method for tbt post-stack inversion
    def tbt_inv(self, mode):
        global m_tbt
        
        epsI = float(epsI_entry.get())
        xl_start, xl_end = xl[0], xl[-1]

        if mode == "2D":

            d_small = data_cube[il_number - il[0], :, int(self.tmin/dt):int(self.tmax/dt)]
            d_small = np.swapaxes(d_small, -1, 0)


            print("\n -------------Running trace-by-trace 2D inversion------------- \n")

            m_tbt, r_tbt = pylops.avo.poststack.PoststackInversion(d_small, wav_est/2, m0 = np.zeros_like(d_small), explicit=True,
                                                                        epsI = epsI, simultaneous = False)
            r_tbt = np.swapaxes(r_tbt, 0, -1)
            m_tbt = np.swapaxes(m_tbt, 0, -1)

            fig_tbt = plt.figure(figsize=(10, 5))
            #fig.subplots_adjust(bottom=0.25)
            gs = GridSpec(2, 2, height_ratios=(10,1))

            ax_seismic = fig_tbt.add_subplot(gs[:1,:])
            ax_histogram = fig_tbt.add_subplot(gs[-1,0])  
            plt.subplots_adjust(left=0.098, right=1, top=0.955, bottom=0.112,hspace=0.25)            
            im = ax_seismic.imshow(m_tbt.T, aspect='auto', cmap='seismic', vmin = 0.1*m_tbt.min(), vmax = 0.1*m_tbt.max(),
                                extent=[xl_start, xl_end, t[int(self.tmax/dt)], t[int(self.tmin/dt)]])
            ax_histogram.hist(m_tbt[0:300,:].T.flatten(), bins = 400)
            ax_histogram.set_title('Histogram of pixel intensities')
            plt.colorbar(im, ax=ax_seismic)

            # Create the RangeSlider
            # Add slider for interactive frame navigation along inline direction
            axframe1 = plt.axes([0.1, 0.01, 0.5, 0.03], facecolor='lightgoldenrodyellow')
            #slider_ax = fig.add_axes([0.20, 0.1, 0.60, 0.03])
            slider = RangeSlider(axframe1, "Threshold", m_tbt.min(), m_tbt.max())

            # Create the Vertical lines on the histogram
            lower_limit_line = ax_histogram.axvline(slider.val[0], color='k')
            upper_limit_line = ax_histogram.axvline(slider.val[1], color='k')
        
        if mode == "3D":

            d_small = data_cube[..., int(self.tmin/dt):int(self.tmax/dt)]
            d_small = np.swapaxes(d_small, -1, 0)

            
            print("\n -------------Running trace-by-trace 3D inversion------------- \n")

            m_tbt, _ = \
                    pylops.avo.poststack.PoststackInversion(d_small, wav_est/2, m0=np.zeros_like(d_small), explicit=True,
                                                                        epsI = epsI, simultaneous = False)
            

            m_tbt = np.swapaxes(m_tbt, 0, -1)
            d_small = np.swapaxes(d_small, 0, -1)

            fig_tbt = plt.figure(figsize=(10, 5))
            #fig.subplots_adjust(bottom=0.25)
            gs = GridSpec(2, 2, height_ratios=(10,1))

            ax_seismic = fig_tbt.add_subplot(gs[:1,:])
            ax_histogram = fig_tbt.add_subplot(gs[-1,0])  
            plt.subplots_adjust(left=0.098, right=1, top=0.955, bottom=0.112,hspace=0.25)            
            im = ax_seismic.imshow(m_tbt[il_number - il[0], ...].T, aspect='auto', cmap='seismic',
                                    vmin = 0.1*m_tbt[il_number - il[0], ...].T.min(),
                                    vmax = 0.1*m_tbt[il_number - il[0], ...].T.max(),
                                extent=[xl_start, xl_end, t[int(self.tmax/dt)], t[int(self.tmin/dt)]])
            ax_histogram.hist(m_tbt[il_number - il[0],...].T.flatten(), bins = 400)
            ax_histogram.set_title('Histogram of pixel intensities')
            plt.colorbar(im, ax=ax_seismic)

            # Create the RangeSlider
            # Add slider for interactive frame navigation along inline direction
            axframe1 = plt.axes([0.1, 0.01, 0.5, 0.03], facecolor='lightgoldenrodyellow')
            #slider_ax = fig.add_axes([0.20, 0.1, 0.60, 0.03])
            slider = RangeSlider(axframe1, "Threshold",
                                m_tbt[il_number - il[0],...].min(),
                                m_tbt[il_number - il[0],...].max())

            # Create the Vertical lines on the histogram
            lower_limit_line = ax_histogram.axvline(slider.val[0], color='k')
            upper_limit_line = ax_histogram.axvline(slider.val[1], color='k')


        def update(val):
            # The val passed to a callback by the RangeSlider will
            # be a tuple of (min, max)

            # Update the image's colormap
            im.norm.vmin = val[0]
            im.norm.vmax = val[1]

            # Update the position of the vertical lines
            lower_limit_line.set_xdata([val[0], val[0]])
            upper_limit_line.set_xdata([val[1], val[1]])

            # Redraw the figure to ensure it updates
            #fig_tbt.canvas.draw_idle()


        slider.on_changed(update)
        plt.grid(False)
        plt.show()

        '''fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        c = ax.imshow(m_tbt.T, aspect='auto', cmap='seismic', vmin = m_tbt.min(), vmax= m_tbt.max(),
                        extent = [xl_start, xl_end, t[int(self.tmax/dt)], t[int(self.tmin/dt)]])
        plt.colorbar(c, ax=ax, pad=0.01)
        plt.grid(False)
        plt.show()'''


        return m_tbt
    # Method for spat reg post-stack inversion
    def spat_inv(self, mode):
        niter_sr = int(niter_sr_entry.get())
        epsI_sr = float(epsI_sr_entry.get())
        epsR_sr = float(epsR_sr_entry.get())

        xl_start, xl_end = xl[0], xl[-1]

        if mode == "2D":
            d_small = data_cube[il_number - il[0], :, int(self.tmin/dt):int(self.tmax/dt)]
            d_small = np.swapaxes(d_small, -1, 0)

            if m_tbt is None:
                m0 = np.zeros_like(d_small)
            else:
                if m_tbt.shape != d_small.shape:
                    m0 = m_tbt[il_number - il[0], ...]
                    m0 = np.swapaxes(m0, -1, 0)
                    print("Adjusted shape of m0:", m0.shape)
                    print("Shape of d_small:", d_small.shape)
                else:
                    m0 = m_tbt
                    print("Shape of m0:", m0.shape)
                    print("Shape of d_small:", d_small.shape)
            print("\n -------------Running spatilly regularized 2D inversion------------- \n")     
            m_relative_reg, r_relative_reg = \
                pylops.avo.poststack.PoststackInversion(d_small, wav_est/2, m0 = m0, epsI = epsI_sr, epsR = epsR_sr, 
                                                    **dict(iter_lim=niter_sr, show=2))
            
            m_relative_reg = np.swapaxes(m_relative_reg, 0, -1)
            r_relative_reg = np.swapaxes(r_relative_reg, 0, -1)
            d_small = np.swapaxes(d_small, 0, -1)
            
            fig = plt.figure(figsize=(10, 5))
            #fig.subplots_adjust(bottom=0.25)
            gs = GridSpec(2, 2, height_ratios=(10,1))

            ax_seismic = fig.add_subplot(gs[:1,:])
            ax_histogram = fig.add_subplot(gs[-1,0])  
            plt.subplots_adjust(left=0.098, right=1, top=0.955, bottom=0.112,hspace=0.25)            
            im = ax_seismic.imshow(m_relative_reg.T, aspect='auto', cmap='seismic', 
                                vmin = 0.1*m_relative_reg.min(),
                                vmax = 0.1*m_relative_reg.max(),
                                extent=[xl_start, xl_end, t[int(self.tmax/dt)], t[int(self.tmin/dt)]])
            ax_histogram.hist(m_relative_reg[0:300,:].T.flatten(), bins = 400)

            ax_histogram.set_title('Histogram of pixel intensities')
            plt.colorbar(im, ax=ax_seismic)

            # Create the RangeSlider
            # Add slider for interactive frame navigation along inline direction
            axframe1 = plt.axes([0.1, 0.01, 0.5, 0.03], facecolor='lightgoldenrodyellow')
            #slider_ax = fig.add_axes([0.20, 0.1, 0.60, 0.03])
            slider = RangeSlider(axframe1, "Threshold",
                                m_relative_reg.min(),
                                m_relative_reg.max())
            
            # Create the Vertical lines on the histogram
            lower_limit_line = ax_histogram.axvline(slider.val[0], color='k')
            upper_limit_line = ax_histogram.axvline(slider.val[1], color='k')


        if mode == "3D":
 
            d_small = data_cube[..., int(self.tmin/dt):int(self.tmax/dt)]
            d_small = np.swapaxes(d_small, -1, 0)

            if m_tbt is None:
                m0 = np.zeros_like(d_small)
            else:
                m0 = m_tbt.T

            print("\n -------------Running spatilly regularized 3D inversion------------- \n")

            m_relative_reg = np.full_like(d_small[:, :, :], np.nan)

            for i in range(len(il)):
                print(f'Running IL {i+1}/{il.shape[0]}\r', end="")
                il_sect = d_small[...,i] 
                m_relative_reg_sect, _ = \
                    pylops.avo.poststack.PoststackInversion(il_sect, wav_est, m0=m0[...,i], epsI=epsI_sr, epsR=epsR_sr, 
                                                            **dict(iter_lim=niter_sr))
                m_relative_reg[...,i] = m_relative_reg_sect

            m_relative_reg = np.swapaxes(m_relative_reg, 0, -1)
            d_small = np.swapaxes(d_small, 0, -1)

            fig = plt.figure(figsize=(10, 5))
            #fig.subplots_adjust(bottom=0.25)
            gs = GridSpec(2, 2, height_ratios=(10,1))

            ax_seismic = fig.add_subplot(gs[:1,:])
            ax_histogram = fig.add_subplot(gs[-1,0])  
            plt.subplots_adjust(left=0.098, right=1, top=0.955, bottom=0.112,hspace=0.25)            
            im = ax_seismic.imshow(m_relative_reg[il_number - il[0], ...].T, aspect='auto', cmap='seismic', 
                                vmin = 0.1*m_relative_reg[il_number - il[0],...].T.min(),
                                vmax = 0.1*m_relative_reg[il_number - il[0],...].T.max(),
                                extent=[xl_start, xl_end, t[int(self.tmax/dt)], t[int(self.tmin/dt)]])
            ax_histogram.hist(m_relative_reg[il_number - il[0],...].T.flatten(), bins = 400)
            ax_histogram.set_title('Histogram of pixel intensities')
            plt.colorbar(im, ax=ax_seismic)

            # Create the RangeSlider
            # Add slider for interactive frame navigation along inline direction
            axframe1 = plt.axes([0.1, 0.01, 0.5, 0.03], facecolor='lightgoldenrodyellow')
            #slider_ax = fig.add_axes([0.20, 0.1, 0.60, 0.03])
            slider = RangeSlider(axframe1, "Threshold",
                                m_relative_reg[il_number - il[0],...].min(),
                                m_relative_reg[il_number - il[0],...].max())
            # Create the Vertical lines on the histogram
            lower_limit_line = ax_histogram.axvline(slider.val[0], color='k')
            upper_limit_line = ax_histogram.axvline(slider.val[1], color='k')

        def update(val):
            # The val passed to a callback by the RangeSlider will
            # be a tuple of (min, max)

            # Update the image's colormap
            im.norm.vmin = val[0]
            im.norm.vmax = val[1]

            # Update the position of the vertical lines
            lower_limit_line.set_xdata([val[0], val[0]])
            upper_limit_line.set_xdata([val[1], val[1]])

            # Redraw the figure to ensure it updates
            #fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.grid(False)
        plt.show()


    # Method for blocky reg post-stack inversion
    def blocky_inv(self, mode):
        niter_b = int(niter_in_b_entry.get())
        niter_out_b = int(niter_out_b_entry.get())
        niter_in_b = int(niter_in_b_entry.get())
        mu_b = float(mu_b_entry.get())
        epsI_b = float(epsI_b_entry.get())
        epsR_b = float(epsR_b_entry.get())
        epsRL1_b = float(epsRL1_b_entry.get())

        il_start = il[0]
        xl_start, xl_end = xl[0], xl[-1]


        if mode == "2D":
            d_small = data_cube[il_number - il[0], :, int(self.tmin/dt) : int(self.tmax/dt)]
            d_small = np.swapaxes(d_small, -1, 0)

            print("\n -------------Running spatially regularized blocky promoting simultaneous 2D inversion------------- \n")

            
            m_blocky, r_blocky = \
                pylops.avo.poststack.PoststackInversion(d_small, wav_est/2, m0 = np.zeros_like(d_small), explicit = False, 
                                                            epsR = epsR_b, epsRL1 = epsRL1_b,
                                                            **dict(mu = mu_b, niter_outer = niter_out_b, 
                                                                niter_inner = niter_in_b, show = True,
                                                                iter_lim = niter_b, damp = epsI_b))
            m_blocky = np.swapaxes(m_blocky, 0, -1)
            r_blocky = np.swapaxes(r_blocky, 0, -1)
            d_small = np.swapaxes(d_small, 0, -1)
            

            fig = plt.figure(figsize=(10, 5))
            #fig.subplots_adjust(bottom=0.25)
            gs = GridSpec(2, 2, height_ratios=(10,1))

            ax_seismic = fig.add_subplot(gs[:1,:])
            ax_histogram = fig.add_subplot(gs[-1,0])  
            plt.subplots_adjust(left=0.098, right=1, top=0.955, bottom=0.112,hspace=0.25)            
            im = ax_seismic.imshow(m_blocky.T, aspect='auto', cmap='seismic', vmin = 0.1*m_blocky.min(), vmax = 0.1*m_blocky.max(),
                                extent=[xl_start, xl_end, t[int(self.tmax/dt)], t[int(self.tmin/dt)]])
            ax_histogram.hist(m_blocky[0:300,:].T.flatten(), bins = 400)
            ax_histogram.set_title('Histogram of pixel intensities')
            plt.colorbar(im, ax=ax_seismic)

            # Create the RangeSlider
            # Add slider for interactive frame navigation along inline direction
            axframe1 = plt.axes([0.1, 0.01, 0.5, 0.03], facecolor='lightgoldenrodyellow')
            #slider_ax = fig.add_axes([0.20, 0.1, 0.60, 0.03])
            slider = RangeSlider(axframe1, "Threshold", m_blocky.min(), m_blocky.max())

            # Create the Vertical lines on the histogram
            lower_limit_line = ax_histogram.axvline(slider.val[0], color='k')
            upper_limit_line = ax_histogram.axvline(slider.val[1], color='k')

        if mode == "3D":
 
            d_small = data_cube[..., int(self.tmin/dt):int(self.tmax/dt)]
            d_small = np.swapaxes(d_small, -1, 0)

            print("\n -------------Running spatially regularized blocky promoting simultaneous 3D inversion------------- \n")

            m_blocky = np.full_like(d_small[:, :, :], np.nan)

            for i in range(len(il)):
                print(f'Running IL {i+1}/{il.shape[0]}\r', end="")
                il_sect = d_small[...,i] 
                m_blocky_sect, _ = \
                    pylops.avo.poststack.PoststackInversion(il_sect, wav_est/2, m0 = np.zeros_like(il_sect), explicit = False, 
                                                            epsR = epsR_b, epsRL1 = epsRL1_b,
                                                            **dict(mu = mu_b, niter_outer = niter_out_b, 
                                                                niter_inner = niter_in_b, show = False,
                                                                iter_lim = niter_b, damp = epsI_b))
                m_blocky[...,i] = m_blocky_sect

            m_blocky = np.swapaxes(m_blocky, 0, -1)
            d_small = np.swapaxes(d_small, 0, -1)

            fig = plt.figure(figsize=(10, 5))
            #fig.subplots_adjust(bottom=0.25)
            gs = GridSpec(2, 2, height_ratios=(10,1))

            ax_seismic = fig.add_subplot(gs[:1,:])
            ax_histogram = fig.add_subplot(gs[-1,0])  
            plt.subplots_adjust(left=0.098, right=1, top=0.955, bottom=0.112,hspace=0.25)            
            im = ax_seismic.imshow(m_blocky[il_number - il[0], ...].T, aspect='auto', cmap='seismic', 
                                vmin = 0.1*m_blocky[il_number - il[0],...].T.min(),
                                vmax = 0.1*m_blocky[il_number - il[0],...].T.max(),
                                extent=[xl_start, xl_end, t[int(self.tmax/dt)], t[int(self.tmin/dt)]])
            ax_histogram.hist(m_blocky[il_number - il[0],...].T.flatten(), bins = 400)
            ax_histogram.set_title('Histogram of pixel intensities')
            plt.colorbar(im, ax=ax_seismic)

            # Create the RangeSlider
            # Add slider for interactive frame navigation along inline direction
            axframe1 = plt.axes([0.1, 0.01, 0.5, 0.03], facecolor='lightgoldenrodyellow')
            #slider_ax = fig.add_axes([0.20, 0.1, 0.60, 0.03])
            slider = RangeSlider(axframe1, "Threshold",
                                m_blocky[il_number - il[0],...].min(),
                                m_blocky[il_number - il[0],...].max())
            # Create the Vertical lines on the histogram
            lower_limit_line = ax_histogram.axvline(slider.val[0], color='k')
            upper_limit_line = ax_histogram.axvline(slider.val[1], color='k')

        def update(val):
            # The val passed to a callback by the RangeSlider will
            # be a tuple of (min, max)

            # Update the image's colormap
            im.norm.vmin = val[0]
            im.norm.vmax = val[1]

            # Update the position of the vertical lines
            lower_limit_line.set_xdata([val[0], val[0]])
            upper_limit_line.set_xdata([val[1], val[1]])

            # Redraw the figure to ensure it updates
            #fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.grid(False)
        plt.show()
        
    # Inversion parameter window
    def inversion_param_wind(self, mode):
        inversion_type = self.inv_type.get()
        if inversion_type == 1:
            self.blocky_inv_param_win(mode)
        if inversion_type == 2:
        # Run regularized spatial inversion
            self.spat_inv_param_win(mode)
        if inversion_type == 3:
        # Run regularized inversion
            self.tbt_inv_param_win(mode)
    # Run inversion buttons    
    def run_inversion(self, mode):
        self.inversion_type = self.inv_type.get()
        if self.inversion_type == 1:
            self.blocky_inv(mode)
        elif self.inversion_type == 2:
        # Run regularized spatial inversion
            self.spat_inv(mode)
        elif self.inversion_type == 3:
        # Run regularized inversion
            self.tbt_inv(mode)
# Object the inherits the function methods and create the GUI application
class Application(Funcs):
    # Python constructor and Tkinter mainloop
    def __init__(self) -> None:
        self.root = Tk()
        self.tela_load()
        self.widgets()
        self.root.mainloop()

    # Tela load method
    def tela_load(self):
        self.root.geometry("550x257")
        self.root.configure(bg = "white")
        self.root.resizable(False, False)

    # Widgets 
    def widgets(self):

        # Create tabs
        self.tabControl = ttk.Notebook(self.root)
        self.tab_load_data = Frame(self.tabControl)
        self.tab_inversion = Frame(self.tabControl)
        
        self.tabControl.add(self.tab_load_data, text='Load Data')
        self.tabControl.add(self.tab_inversion, text='Inversion')
        self.tabControl.pack(expand=1, fill='both')
        
        #canvas for data loading
        self.canvas_load = Canvas(
            self.tab_load_data,
            bg = "white",
            height = 257,
            width = 550,
            bd = 0,
            highlightthickness = 0,
            relief = "ridge"
        )

        self.canvas_load.pack(expand=True, fill="both")
        self.canvas_load.place(x = 0, y = 0)
        self.canvas_load.pack(expand = True)

        
        self.canvas_load.create_text(
            180.0,
            20.0,
            text="Enter IL:",
            fill="black",
            font=("Inter", 12 * -1)
        )

        self.il_entry = Entry(
                        self.tab_load_data,
                        bg="lightgrey")
        
        self.il_entry.place(
            x=159.0,
            y=30.0,
            width=116.0,
            height=24.0
        )
        
        self.canvas_inv = Canvas(
            self.tab_load_data,
            bg = "white",
            height = 257,
            width = 550,
            bd = 0,
            highlightthickness = 0,
            relief = "ridge"
        )

        self.button_image_disp = PhotoImage(
            file=os.path.join(ASSETS_PATH,"button_disp.png"))
        button_disp = Button(
            self.tab_load_data,
            image=self.button_image_disp,
            borderwidth=0,
            highlightthickness=0,
            command=self.get_iline_num,
            relief="flat"
        )

        button_disp.place(
            x=159.0,
            y=68.0,
            width=116.0,
            height=24.0
        )

        self.button_image_load = PhotoImage(
            file=os.path.join(ASSETS_PATH, "button_load.png"))
        self.button_load = Button(
            self.tab_load_data,
            image=self.button_image_load,
            borderwidth=0,
            highlightthickness=0,
            command=self.load_file,
            relief="flat"
        )

        self.button_load.place(
            x=15.0,
            y=13.0,
            width=116.0,
            height=24.0
        )

        self.button_image_close = PhotoImage(
            file=os.path.join(ASSETS_PATH, "button_close.png"))
        button_close = Button(
            self.tab_load_data,
            image=self.button_image_close,
            borderwidth=0,
            highlightthickness=0,
            command=self.root.destroy,
            relief="flat"
        )

        button_close.place(
            x=15.0,
            y=53.0,
            width=116.0,
            height=24.0
        )

        self.image_logo = PhotoImage(
            file=os.path.join(ASSETS_PATH,"image_1.png"))
        image_1 = self.canvas_load.create_image(
            470.0,
            50.0,
            image=self.image_logo
        )

        self.il_label = Label(self.tab_load_data,
                        bg="white"
        )
        self.il_label.place(
            x=280.0,
            y=38.0
        )
        self.status_label = Label(self.tab_load_data,
                            bg="white")
        self.status_label.place(x=25.0,
                            y=200)

        #canvas for inversion
        self.canvas_inv = Canvas(
            self.tab_inversion,
            bg = "white",
            height = 257,
            width = 550,
            bd = 0,
            highlightthickness = 0,
            relief = "ridge"
        )

        self.canvas_inv.pack(expand=True, fill="both")
        self.canvas_inv.place(x = 0, y = 0)
        self.canvas_inv.pack(expand = True)

        image_1 = self.canvas_inv.create_image(
            470.0,
            50.0,
            image=self.image_logo
        )
                  
        self.button_image_tied_wav = PhotoImage(
            file=os.path.join(ASSETS_PATH, "button_tied_wvl.png"))
        
        button_load_wav = Button(
            self.tab_inversion,
            image = self.button_image_tied_wav,
            borderwidth=0,
            highlightthickness=0,
            command=self.input_tied_wvlt,
            relief="flat"
        )

        button_load_wav.place(
            x=15.0,
            y=13.0,
            width=116.0,
            height=24.0
        )

        self.button_image_wav = PhotoImage(
            file=os.path.join(ASSETS_PATH, "button_wvl.png"))
        
        button_wav = Button(
            self.tab_inversion,
            image=self.button_image_wav,
            borderwidth=0,
            highlightthickness=0,
            command=self.open_parameter_window,
            relief="flat"
        )

        button_wav.place(
            x=15.0,
            y=53.0,
            width=116.0,
            height=24.0
        )

        # Radio buttons for inversion type
        self.inv_type = IntVar(self.tab_inversion)
        self.inv_type.set(3)  # Default to Blocky Inversion
        self.blocky_inv_img = PhotoImage(
                        file=os.path.join(ASSETS_PATH,"blocky.png"))
        blocky_radio = Radiobutton(self.tab_inversion,
                                image=self.blocky_inv_img,
                                text="Blocky Inversion", 
                                variable=self.inv_type, 
                                value=1,
                                relief="flat", 
                                borderwidth=0, 
                                highlightthickness=0)
        blocky_radio.place(x=400, 
                        y=120)
        self.reg_spt_img = PhotoImage(
                        file=os.path.join(ASSETS_PATH, "reg.png"))
        reg_spatial_radio = Radiobutton(self.tab_inversion,
                                        image=self.reg_spt_img, 
                                        text="Regularized Spatial Inversion", 
                                        variable=self.inv_type, value=2, 
                                        relief="flat", 
                                        borderwidth=0, 
                                        highlightthickness=0)
        reg_spatial_radio.place(x=400, 
                                y=150)
        
        self.tbt_img = PhotoImage(
                        file=os.path.join(ASSETS_PATH,"tbt.png"))
        
        reg_radio = Radiobutton(self.tab_inversion,
                                image=self.tbt_img,
                                text="Trace-by-trace Inversion", 
                                variable=self.inv_type, value=3, 
                                relief="flat", 
                                borderwidth=0, 
                                highlightthickness=0)
        reg_radio.place(x=400, 
                        y=180)

        # Buttons for 2D and 3D inversion
        self.run_2D_button_img = PhotoImage(file=os.path.join(ASSETS_PATH, "run_2D.png"))
        run_2D_button = Button(
            self.tab_inversion, 
            image=self.run_2D_button_img,
            text="Open inversion params", 
            command = lambda: self.inversion_param_wind("2D"), 
            relief="flat", 
            borderwidth=0, 
            highlightthickness=0
        )
        run_2D_button.place(x=13.0, y=93.0, width=116.0, height=24.0)
        
        self.run_3D_button_img = PhotoImage(file=os.path.join(ASSETS_PATH, "run_3D.png"))
        run_3D_button = Button(
            self.tab_inversion, 
            image=self.run_3D_button_img,
            text="Open inversion params", 
            command=lambda: self.inversion_param_wind("3D"), 
            relief="flat", 
            borderwidth=0, 
            highlightthickness=0
        )
        run_3D_button.place(x=13.0, y=133.0, width=116.0, height=24.0)
        
#call aplication
Application()