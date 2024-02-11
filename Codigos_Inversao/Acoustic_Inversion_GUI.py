import segyio
import numpy as np
import matplotlib.pyplot as plt
import pylops
from tkinter import filedialog
from tkinter import *

# Global variables
data_cube = None
il = None
xl = None
t = None
wav_est = None
m_tbt = None

class Funcs():
    # Function to load SEGY file
    def load_file(self):
        global data_cube, il, xl, t
        filename = filedialog.askopenfilename()
        if filename:
            with segyio.open(filename, iline=189, xline=193) as stack:
                ils, xls, twt = stack.ilines, stack.xlines, stack.samples
                data_cube = segyio.cube(stack)
            il, xl, t = ils, xls, twt
            self.status_label.config(text=f"File loaded: {filename}")
            # Enable the button to open the parameter input window
            self.parameter_button.config(state=NORMAL)
            print("File loaded successfully.")
            return data_cube, il, xl, t
        else:
            return None, None, None, None

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

        estimate_button = Button(parameter_window, text="Estimate Wavelet", command=self.estimate_wavelet)
        estimate_button.grid(row=3,column=1)
        return self.tmin_entry, self.tmax_entry


    # Function for wavelet estimation
    def estimate_wavelet(self):
        global data_cube, t, wav_est
        self.tmin = float(self.tmin_entry.get())
        self.tmax = float(self.tmax_entry.get())
        nt_wav = 16
        nfft = 2**8
        dt = t[1] - t[0]
        wav_est = np.mean(np.abs(np.fft.fft(data_cube[..., int(self.tmin/dt):int(self.tmax/dt)], nfft, axis=-1)), axis=(0, 1))
        wav_est = np.real(np.fft.ifft(wav_est)[:nt_wav])
        wav_est = np.concatenate((np.flipud(wav_est[1:]), wav_est), axis=0)
        wav_est = wav_est / wav_est.max()
        wcenter = np.argmax(np.abs(wav_est))
        fig, axs = plt.subplots(1, 2, figsize=(20, 5))
        fig.suptitle('Statistical wavelet estimate')
        axs[0].plot(wav_est, 'k')
        axs[0].set_title('Time')
        plt.show()
        return wav_est

    # Function for post-stack inversion
    def tbt_inv(self):
        global data_cube, il, t, wav_est, m_tbt

        il_start = il[0]
        xl_start, xl_end = xl[0], xl[-1]
        dt = t[1] - t[0]
        d_small = data_cube[..., int(self.tmin/dt):int(self.tmax/dt)]
        d_small = np.swapaxes(d_small, -1, 0)
        
        print("\n -------------Running trace-by-trace inversion------------- \n")
        
        m_tbt, r_tbt = pylops.avo.poststack.PoststackInversion(d_small, wav_est/2, m0=np.zeros_like(d_small), explicit=True,
                                                                    epsI=1e-3, simultaneous=False)
        m_tbt = np.swapaxes(m_tbt, 0, -1)
        r_tbt = np.swapaxes(r_tbt, 0, -1)
        d_small = np.swapaxes(d_small, 0, -1)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        c = ax.imshow(m_tbt[self.il_number - il_start, :, :].T, aspect='auto', cmap='seismic', vmin=-0.1*m_tbt.max(), vmax=0.1*m_tbt.max(),
                    extent=[xl_start, xl_end, t[int(self.tmax/dt)], t[int(self.tmin/dt)]])
        plt.colorbar(c, ax=ax, pad=0.01)
        plt.grid(False)
        plt.show()
        return m_tbt
    # Function for post-stack inversion
    def spat_reg(self):
        global data_cube, il, t, wav_est, m_tbt
        il_start = il[0]
        xl_start, xl_end = xl[0], xl[-1]
        dt = t[1] - t[0]
        d_small = data_cube[..., int(self.tmin/dt):int(self.tmax/dt)]
        d_small = np.swapaxes(d_small, -1, 0)
        #-----------------------------------------------------------------------------------------------------
        # Spatially simultaneous
        niter_sr = 10 # number of iterations of lsqr
        epsI_sr = 1e-4 # damping
        epsR_sr = 1e2 # spatial regularization
        


        print("\n -------------Running spatially regularized simultaneous inversion------------- \n")
        
        if m_tbt is None:
            m0 = np.zeros_like(d_small)
        else:
            print("2")
            m0 = m_tbt.T
        
        m_relative_reg, r_relative_reg = \
            pylops.avo.poststack.PoststackInversion(d_small, wav_est/2, m0=m0, epsI=epsI_sr, epsR=epsR_sr, 
                                            **dict(iter_lim=niter_sr, show=2))

        m_relative_reg = np.swapaxes(m_relative_reg, 0, -1)
        r_relative_reg = np.swapaxes(r_relative_reg, 0, -1)
        d_small = np.swapaxes(d_small, 0, -1)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        c = ax.imshow(m_relative_reg[self.il_number - il_start, :, :].T, aspect='auto', cmap='seismic', vmin=-0.1*m_relative_reg.max(), vmax=0.1*m_relative_reg.max(),
                    extent=[xl_start, xl_end, t[int(self.tmax/dt)], t[int(self.tmin/dt)]])
        plt.colorbar(c, ax=ax, pad=0.01)
        plt.grid(False)
        plt.show()

    # Function for post-stack inversion
    def blocky_inv(self):
        global data_cube, il, t, wav_est

        il_start = il[0]
        xl_start, xl_end = xl[0], xl[-1]
        dt = t[1] - t[0]
        d_small = data_cube[..., int(self.tmin/dt):int(self.tmax/dt)]
        d_small = np.swapaxes(d_small, -1, 0)
        #-----------------------------------------------------------------------------------------------------
        # Blocky simultaneous
        niter_out_b = 3 # number of outer loop iterations
        niter_in_b = 1 # number of inner loop iterations
        niter_b = 10 # number of iterations of lsqr
        mu_b = 1e-1 # damping for data term
        epsI_b = 1e-4 # damping
        epsR_b = 0.1 # spatial regularization
        epsRL1_b = 0.2 # blocky regularization
        #-----------------------------------------------------------------------------------------------------

        print("\n -------------Running spatially regularized blocky promoting simultaneous inversion------------- \n")
        
        #self.d_small = np.swapaxes(self.d_small, -1, 0)

        m_blocky, r_blocky = \
            pylops.avo.poststack.PoststackInversion(d_small, wav_est/2, m0=np.zeros_like(d_small), explicit=False, 
                                                    epsR=epsR_b, epsRL1=epsRL1_b,
                                                    **dict(mu=mu_b, niter_outer=niter_out_b, 
                                                        niter_inner=niter_in_b, show=True,
                                                        iter_lim=niter_b, damp=epsI_b))
        m_blocky = np.swapaxes(m_blocky, 0, -1)
        r_blocky = np.swapaxes(r_blocky, 0, -1)
        d_small = np.swapaxes(d_small, 0, -1)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        c = ax.imshow(m_blocky[self.il_number - il_start, :, :].T, aspect='auto', cmap='seismic', vmin=-0.1*m_blocky.max(), vmax=0.1*m_blocky.max(),
                    extent=[xl_start, xl_end, t[int(self.tmax/dt)], t[int(self.tmin/dt)]])
        plt.colorbar(c, ax=ax, pad=0.01)
        plt.grid(False)
        plt.show()

    # Function to get IL number from input
    def get_iline_num(self):
        global data_cube, il, t, wav_est
        try:
            self.il_number = int(self.il_entry.get())
            self.il_label.config(text=f"Selected IL: {self.il_number}")

            il_start= il[0]
            xl_start, xl_end = xl[0], xl[-1]
            dt = t[1] - t[0]

            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            c = ax.imshow(data_cube[self.il_number - il_start, :, int(self.tmin/dt):int(self.tmax/dt)].T, aspect='auto', cmap='seismic', vmin=-0.1*data_cube.max(), vmax=0.1*data_cube.max(),
                        extent=[xl_start, xl_end, t[int(self.tmax/dt)], t[int(self.tmin/dt)]])
            plt.colorbar(c, ax=ax, pad=0.01)
            plt.grid(False)
            plt.show()                  
            return self.il_number
        except ValueError:
            self.status_label.config(text="Error: Please enter valid IL number.")
            return None
        
    def run_inversion(self):
            inversion_type = self.inversion_type.get()
            if inversion_type == 1:
                self.blocky_inv()
            elif inversion_type == 2:
                # Run regularized spatial inversion
                self.spat_reg()
            elif inversion_type == 3:
                # Run regularized inversion
                self.tbt_inv()

class Application(Funcs):
    def __init__(self) -> None:
        self.root = Tk()
        self.tela_load()
        self.frames()
        self.widgets_frame1()
        self.root.mainloop()

    def tela_load(self):
        self.root.title("File Loader")
        self.root.configure(background='lightblue')
        self.root.geometry("600x400")
        self.root.resizable(True, True)
    def frames(self):
        self.frame_1 = Frame(self.root)
        self.frame_1.place(relx=0.02, rely=0.02, relwidth= 0.96, relheight= 0.5)   
    def widgets_frame1(self):


        # Create widgets
        load_button = Button(self.frame_1, text="Load File", command=self.load_file)
        load_button.place(relx=0.1, rely=0.1, relwidth= 0.2, relheight= 0.1)

        run_button = Button(self.frame_1, text="Run inversion", command=self.run_inversion)
        run_button.place(relx=0.8, rely=0.3, relwidth=0.2, relheight=0.1)

        il_number_label = Label(self.frame_1, text="Enter IL number for display")
        il_number_label.place(relx=0.35, rely=0.0, relwidth= 0.3, relheight= 0.1)
        self.il_entry = Entry(self.frame_1)
        self.il_entry.place(relx=0.4, rely=0.1, relwidth= 0.25, relheight= 0.1)       
        il_entry_bt = Button(self.frame_1, text="Display IL", command=self.get_iline_num)
        il_entry_bt.place(relx=0.4, rely=0.2, relwidth= 0.2, relheight= 0.1)
        self.il_label = Label(self.frame_1, text="")
        self.il_label.place(relx=0.6, rely=0.1, relwidth= 0.2, relheight= 0.1)

        open_button2 = Button(self.frame_1, text="Close the window", command=self.root.destroy)
        open_button2.place(relx=0.1, rely=0.3, relwidth= 0.2, relheight= 0.1)

        # Button to open the parameter input window (disabled initially)
        self.parameter_button = Button(self.frame_1, text="Enter parameters", command=self.open_parameter_window, state=DISABLED)
        self.parameter_button.place(relx=0.1, rely=0.2, relwidth= 0.2, relheight= 0.1)

        # Status label
        self.status_label = Label(self.frame_1, text="")
        self.status_label.place(relx=0.0, rely=0.9, relwidth= 1, relheight= 0.1)
        # Radio buttons for inversion type
        self.inversion_type = IntVar(self.root)
        self.inversion_type.set(1)  # Default to Blocky Inversion

        self.blocky_radio = Radiobutton(self.frame_1, text="Blocky Inversion", variable=self.inversion_type, value=1)
        self.blocky_radio.place(relx=0.6, rely=0.3)

        self.reg_spatial_radio = Radiobutton(self.frame_1, text="Regularized Spatial Inversion", variable=self.inversion_type, value=2)
        self.reg_spatial_radio.place(relx=0.6, rely=0.4)

        self.reg_radio = Radiobutton(self.frame_1, text="Trace-by-trace Inversion", variable=self.inversion_type, value=3)
        self.reg_radio.place(relx=0.6, rely=0.5)

# Run the Tkinter event loop
Application()
