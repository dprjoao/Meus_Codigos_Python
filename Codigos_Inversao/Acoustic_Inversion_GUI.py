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
            self.parameter_button.config(state=NORMAL)
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
                fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                c = ax.imshow(data_cube[..., int(4000/dt)], aspect='auto', cmap='gray_r', vmin=-0.1*data_cube.max(), vmax=0.1*data_cube.max(),
                            extent=[xl_start, xl_end, il_start, il_end])
                plt.colorbar(c, ax=ax, pad=0.01)
                plt.grid(False)
                plt.show()
            else:
                pass           
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
        
        close_button = Button(parameter_window, text="Close", command= parameter_window.destroy)
        close_button.grid(row=4,column=1)
        
        return self.tmin_entry, self.tmax_entry

    def tbt_inv_param_win(self):
        inv_param_win = Toplevel(self.root)
        inv_param_win.title("Parameter Input")

        epsI_label = Label(inv_param_win, text="Damping (epsI):")
        epsI_label.grid(row=0, column=0)
        self.epsI_entry = Entry(inv_param_win)
        self.epsI_entry.grid(row=0, column=1)

        run_inv_button = Button(inv_param_win, text="Run inversion", command=self.run_inversion)
        run_inv_button.grid(row=1,column=1)
        
        close_button = Button(inv_param_win, text="Close", command= inv_param_win.destroy)
        close_button.grid(row=2,column=1)


    def spat_inv_param_win(self):
        inv_param_win = Toplevel(self.root)
        inv_param_win.title("Parameter Input")

        # Create labels and entry widgets for parameter input
        niter_label = Label(inv_param_win, text="Number of iterations:")
        niter_label.grid(row=0, column=0)
        self.niter_sr_entry = Entry(inv_param_win)
        self.niter_sr_entry.grid(row=0, column=1)

        epsI_label = Label(inv_param_win, text="Damping (epsI):")
        epsI_label.grid(row=1, column=0)
        self.epsI_sr_entry = Entry(inv_param_win)
        self.epsI_sr_entry.grid(row=1, column=1)

        epsR_label = Label(inv_param_win, text="Spatial regularization (epsR):")
        epsR_label.grid(row=2, column=0)
        self.epsR_sr_entry = Entry(inv_param_win)
        self.epsR_sr_entry.grid(row=2, column=1)


        run_inv_button = Button(inv_param_win, text="Run inversion", command=self.run_inversion)
        run_inv_button.grid(row=3,column=1)
        
        close_button = Button(inv_param_win, text="Close", command= inv_param_win.destroy)
        close_button.grid(row=4,column=1)

    def blocky_inv_param_win(self):
        inv_param_win = Toplevel(self.root)
        inv_param_win.title("Parameter Input")

        # Create labels and entry widgets for parameter input
        niter_out_label = Label(inv_param_win, text="Number of outer loop iterations:")
        niter_out_label.grid(row=0, column=0)
        self.niter_out_b_entry = Entry(inv_param_win)
        self.niter_out_b_entry.grid(row=0, column=1)

        niter_in_b_label = Label(inv_param_win, text="Number of inner loop iterations:")
        niter_in_b_label.grid(row=1, column=0)
        self.niter_in_b_entry = Entry(inv_param_win)
        self.niter_in_b_entry.grid(row=1, column=1)

        niter_b_label = Label(inv_param_win, text="Number of iterations of lsqr:")
        niter_b_label.grid(row=2, column=0)
        self.niter_b_entry = Entry(inv_param_win)
        self.niter_b_entry.grid(row=2, column=1)

        mu_b_label = Label(inv_param_win, text="Data term damping:")
        mu_b_label.grid(row=3, column=0)
        self.mu_b_entry = Entry(inv_param_win)
        self.mu_b_entry.grid(row=3, column=1)

        epsI_b_label = Label(inv_param_win, text="Damping:")
        epsI_b_label.grid(row=4, column=0)
        self.epsI_b_entry = Entry(inv_param_win)
        self.epsI_b_entry.grid(row=4, column=1)

        epsR_b_label = Label(inv_param_win, text="Spatial regularization:")
        epsR_b_label.grid(row=5, column=0)
        self.epsR_b_entry = Entry(inv_param_win)
        self.epsR_b_entry.grid(row=5, column=1)

        epsRL1_b_label = Label(inv_param_win, text="Blocky regularization:")
        epsRL1_b_label.grid(row=6, column=0)
        self.epsRL1_b_entry = Entry(inv_param_win)
        self.epsRL1_b_entry.grid(row=6, column=1)

        run_inv_button = Button(inv_param_win, text="Run inversion", command=self.run_inversion)
        run_inv_button.grid(row=7,column=1)
        
        close_button = Button(inv_param_win, text="Close", command= inv_param_win.destroy)
        close_button.grid(row=8,column=1)

    # Function for wavelet estimation
    def estimate_wavelet(self):
        global data_cube, t, wav_est
        self.tmin = float(self.tmin_entry.get())
        self.tmax = float(self.tmax_entry.get())
        nt_wav = 16
        nfft = 2**8
        dt = t[1] - t[0]
        wav_est_fft = np.mean(np.abs(np.fft.fft(data_cube[..., int(self.tmin/dt):int(self.tmax/dt)], nfft, axis=-1)), axis=(0, 1))
        fwest = np.fft.fftfreq(nfft, d=dt/1000)
        wav_est = np.real(np.fft.ifft(wav_est_fft)[:nt_wav])
        wav_est = np.concatenate((np.flipud(wav_est[1:]), wav_est), axis=0)
        wav_est = wav_est / wav_est.max()
        # display wavelet
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle('Statistical wavelet estimate')
        axs[0].plot(fwest[:nfft//2], wav_est_fft[:nfft//2], 'k')
        axs[0].set_title('Frequency')
        axs[1].plot(wav_est, 'k')
        axs[1].set_title('Time');
        plt.show()
        return wav_est

    # Function for post-stack inversion
    def tbt_inv(self):
        global data_cube, il, t, wav_est, m_tbt
        self.epsI = float(self.epsI_entry.get())
        il_start = il[0]
        xl_start, xl_end = xl[0], xl[-1]
        dt = t[1] - t[0]
        d_small = data_cube[self.il_number - il_start, :, int(self.tmin/dt):int(self.tmax/dt)]
        d_small = np.swapaxes(d_small, -1, 0)
        
        print("\n -------------Running trace-by-trace inversion------------- \n")
        
        m_tbt, r_tbt = pylops.avo.poststack.PoststackInversion(d_small, wav_est/2, m0=np.zeros_like(d_small), explicit=True,
                                                                    epsI=self.epsI, simultaneous=False)
        m_tbt = np.swapaxes(m_tbt, 0, -1)
        r_tbt = np.swapaxes(r_tbt, 0, -1)
        d_small = np.swapaxes(d_small, 0, -1)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        c = ax.imshow(m_tbt.T, aspect='auto', cmap='seismic', vmin=-0.1*m_tbt.max(), vmax=0.1*m_tbt.max(),
                    extent=[xl_start, xl_end, t[int(self.tmax/dt)], t[int(self.tmin/dt)]])
        plt.colorbar(c, ax=ax, pad=0.01)
        plt.grid(False)
        plt.show()
        return m_tbt
    
    # Function for post-stack inversion
    def spat_inv(self):
        global data_cube, il, t, wav_est, m_tbt
        self.niter_sr = int(self.niter_sr_entry.get())
        self.epsI_sr = float(self.epsI_sr_entry.get())
        self.epsR_sr = float(self.epsR_sr_entry.get())

        il_start = il[0]
        xl_start, xl_end = xl[0], xl[-1]
        dt = t[1] - t[0]
        d_small = data_cube[self.il_number - il_start, :, int(self.tmin/dt):int(self.tmax/dt)]
        d_small = np.swapaxes(d_small, -1, 0)

        print("\n -------------Running spatially regularized simultaneous inversion------------- \n")
        
        if m_tbt is None:
            m0 = np.zeros_like(d_small)
        else:
            print("2")
            m0 = m_tbt.T
        
        m_relative_reg, r_relative_reg = \
            pylops.avo.poststack.PoststackInversion(d_small, wav_est/2, m0=m0, epsI=self.epsI_sr, epsR=self.epsR_sr, 
                                            **dict(iter_lim=self.niter_sr, show=2))

        m_relative_reg = np.swapaxes(m_relative_reg, 0, -1)
        r_relative_reg = np.swapaxes(r_relative_reg, 0, -1)
        d_small = np.swapaxes(d_small, 0, -1)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        c = ax.imshow(m_relative_reg.T, aspect='auto', cmap='seismic', vmin=-0.1*m_relative_reg.max(), vmax=0.1*m_relative_reg.max(),
                    extent=[xl_start, xl_end, t[int(self.tmax/dt)], t[int(self.tmin/dt)]])
        plt.colorbar(c, ax=ax, pad=0.01)
        plt.grid(False)
        plt.show()

    # Function for post-stack inversion
    def blocky_inv(self):
        global data_cube, il, t, wav_est

        self.niter_b = int(self.niter_b_entry.get())
        self.niter_out_b = int(self.niter_out_b_entry.get())
        self.niter_in_b = int(self.niter_in_b_entry.get())
        self.mu_b = float(self.mu_b_entry.get())
        self.epsI_b = float(self.epsI_b_entry.get())
        self.epsR_b = float(self.epsR_b_entry.get())
        self.epsRL1_b = float(self.epsRL1_b_entry.get())

        il_start = il[0]
        xl_start, xl_end = xl[0], xl[-1]
        dt = t[1] - t[0]
        d_small = data_cube[self.il_number - il_start, :, int(self.tmin/dt):int(self.tmax/dt)]
        d_small = np.swapaxes(d_small, -1, 0)

        print("\n -------------Running spatially regularized blocky promoting simultaneous inversion------------- \n")
        
        m_blocky, r_blocky = \
            pylops.avo.poststack.PoststackInversion(d_small, wav_est/2, m0=np.zeros_like(d_small), explicit=False, 
                                                    epsR=self.epsR_b, epsRL1=self.epsRL1_b,
                                                    **dict(mu=self.mu_b, niter_outer=self.niter_out_b, 
                                                        niter_inner=self.niter_in_b, show=True,
                                                        iter_lim=self.niter_b, damp=self.epsI_b))
        m_blocky = np.swapaxes(m_blocky, 0, -1)
        r_blocky = np.swapaxes(r_blocky, 0, -1)
        d_small = np.swapaxes(d_small, 0, -1)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        c = ax.imshow(m_blocky.T, aspect='auto', cmap='seismic', vmin=-0.1*m_blocky.max(), vmax=0.1*m_blocky.max(),
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
            c = ax.imshow(data_cube[self.il_number - il_start, :, int(self.tmin/dt):int(self.tmax/dt)].T, aspect='auto', cmap='gray_r', vmin=-0.1*data_cube.max(), vmax=0.1*data_cube.max(),
                        extent=[xl_start, xl_end, t[int(self.tmax/dt)], t[int(self.tmin/dt)]])
            plt.colorbar(c, ax=ax, pad=0.01)
            plt.grid(False)
            plt.show()                  
            return self.il_number
        except ValueError:
            self.status_label.config(text="Error: Please enter valid IL number.")
            return None
   
    def inversion_param_wind(self):
            inversion_type = self.inversion_type.get()
            if inversion_type == 1:
                self.blocky_inv_param_win()
            elif inversion_type == 2:
                # Run regularized spatial inversion
                self.spat_inv_param_win()
            elif inversion_type == 3:
                # Run regularized inversion
                self.tbt_inv_param_win()
           
    def run_inversion(self):
            inversion_type = self.inversion_type.get()
            if inversion_type == 1:
                self.blocky_inv()
            elif inversion_type == 2:
                # Run regularized spatial inversion
                self.spat_inv()
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
        self.root.configure(background="#4281A4")
        self.root.geometry("600x400")
        self.root.resizable(True, True)
    def frames(self):
        self.frame_1 = Frame(self.root)
        self.frame_1.place(relx=0.02, rely=0.02, relwidth= 0.96, relheight= 0.5)   
    def widgets_frame1(self):


        # Create widgets
        load_button = Button(self.frame_1, text="Load File", command=self.load_file)
        load_button.place(relx=0.01, rely=0.01, relwidth= 0.2, relheight= 0.1)

        # Button to open the parameter input window (disabled initially)
        self.parameter_button = Button(self.frame_1, text="Enter parameters", command=self.open_parameter_window, state=DISABLED)
        self.parameter_button.place(relx=0.01, rely=0.11, relwidth= 0.2, relheight= 0.1)

        #Button to close aplication window
        close_button2 = Button(self.frame_1, text="Close the window", command=self.root.destroy)
        close_button2.place(relx=0.01, rely=0.21, relwidth= 0.2, relheight= 0.1)

        #Run inversion button
        run_button = Button(self.frame_1, text="Open inversion params", command=self.inversion_param_wind)
        run_button.place(relx=0.8, rely=0.3, relwidth=0.2, relheight=0.1)
        
        #Inline display and buttons
        il_number_label = Label(self.frame_1, text="Enter IL number for display")
        il_number_label.place(relx=0.2, rely=0.0, relwidth= 0.3, relheight= 0.1)
        self.il_entry = Entry(self.frame_1)
        self.il_entry.place(relx=0.25, rely=0.1, relwidth= 0.2, relheight= 0.1)       
        il_entry_bt = Button(self.frame_1, text="Display IL", command=self.get_iline_num)
        il_entry_bt.place(relx=0.25, rely=0.2, relwidth= 0.2, relheight= 0.1)
        self.il_label = Label(self.frame_1, text="")
        self.il_label.place(relx=0.45, rely=0.1, relwidth= 0.2, relheight= 0.1)

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
