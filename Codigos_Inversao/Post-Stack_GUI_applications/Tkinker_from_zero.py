from tkinter import filedialog
from tkinter import *
import segyio
import matplotlib.pyplot as plt
import numpy as np
import pylops
class Funcs():
    def load_file(self):
        self.filename = filedialog.askopenfilename()
        if self.filename:
            # Enable the button to open the parameter input window
            self.parameter_button.config(state=NORMAL)
            self.file_label.config(text=f"File loaded: {self.filename}")
            # Store the filename in a global variable or pass it to another function as needed
            self.selected_file = self.filename
            #Loading seismic using Segyio lib
            # string containing the path location of the seismic data at disk
            self.segy_file_path = self.selected_file    #<-----------------Nome e caminho para a sismica
            #loading stack using Segyio lib
            with segyio.open(self.segy_file_path,iline=189, xline=193) as self.stack:
                #Allocating IL, XL, Time axis in variables
                self.ils, self.xls, self.twt = self.stack.ilines, self.stack.xlines, self.stack.samples
                #Creating seismic cube format using segyio cube method
                self.data = segyio.cube(self.stack)
                print("File loaded successfully.")
    def get_iline_num(self):
        self.il_num =  int(self.il_entry.get())
        self.status_label.config(text=f"Selected IL: {self.il_num}")

        self.nil, self.nxl, self.nt = self.data.shape
        self.il_start, self.il_end = self.ils[0], self.ils[-1]
        self.xl_start, self.xl_end = self.xls[0], self.xls[-1]
        self.dt = self.twt[1] - self.twt[0]

        print("SEG-Y file information:")
        print(f"Data shape: {self.nil} ILs x {self.nxl} XLs x {self.nt} Time samples")
        print(f"IL range: {self.il_start}-{self.il_end}")
        print(f"XL range: {self.xl_start}-{self.xl_end}")
        print(f"Time sample interval: {self.dt} ms")
        #plotting data
        fig, ax = plt.subplots(figsize=(10, 5))

        c=ax.imshow(self.data[self.il_num-self.il_start, :, :].T, aspect='auto', cmap='gray_r', vmin=-3000, vmax=3000,
                    extent=[self.xl_start, self.xl_end, self.twt[-1], self.twt[0]])

        plt.colorbar(c, ax=ax, pad=0.01)
        plt.grid(False)
        ax.set_ylim(6000, 4500)
        plt.show()
    
    def open_parameter_window(self):
        self.parameter_window = Toplevel(self.root)
        self.parameter_window.title("Parameter Input")

        # Create labels and entry widgets for parameter input
        tmin_label = Label(self.parameter_window, text="Enter Time min in ms:")
        tmin_label.grid(row=0, column=0)

        tmin_entry = Entry(self.parameter_window)
        tmin_entry.grid(row=0, column=1)

        tmax_label = Label(self.parameter_window, text="Enter Time max in ms:")
        tmax_label.grid(row=1, column=0)

        tmax_entry = Entry(self.parameter_window)
        tmax_entry.grid(row=1, column=1)

        # Button to trigger computation
        compute_button = Button(self.parameter_window, text="Compute Wavelet", command=lambda: self.compute_wvlt(tmin_entry, tmax_entry))
        compute_button.grid(row=2, columnspan=2)

    def compute_wvlt(self, tmin_entry, tmax_entry):
        try:
            self.t_min = float(tmin_entry.get())
            self.t_max = float(tmax_entry.get())
            # Call the wavelet computation method
            self.calculate_wavelet()
        except ValueError:
            print("Invalid input for time min/max. Please enter numeric values.")


    def calculate_wavelet(self):
        self.dt = self.twt[1] - self.twt[0]
        #Another wavelet estimation method
        self.d_small = self.data[..., int(self.t_min/self.dt):int(self.t_max/self.dt)]
        nt_wav = 16 # lenght of wavelet in samples
        nfft = 2**8 # lenght of fft

        # time axis for wavelet
        t_wav = np.arange(nt_wav) * (self.dt/1000) 
        t_wav = np.concatenate((np.flipud(-t_wav[1:]), t_wav), axis=0)

        # estimate wavelet spectrum
        wav_est_fft = np.mean(np.abs(np.fft.fft(self.d_small[..., :], nfft, axis=-1)), axis=(0, 1))#<-----------------Janela da FFT em indice
        fwest = np.fft.fftfreq(nfft, d=self.dt/1000)

        # create wavelet in time
        self.wav_est = np.real(np.fft.ifft(wav_est_fft)[:nt_wav])
        self.wav_est = np.concatenate((np.flipud(self.wav_est[1:]), self.wav_est), axis=0)
        wav_est = self.wav_est / self.wav_est.max()
        wcenter = np.argmax(np.abs(wav_est))

        # display wavelet
        fig, axs = plt.subplots(1, 2, figsize=(20, 5))
        fig.suptitle('Statistical wavelet estimate')
        axs[0].plot(fwest[:nfft//2], wav_est_fft[:nfft//2], 'k')
        axs[0].set_title('Frequency')
        axs[1].plot(self.wav_est, 'k')
        axs[1].set_title('Time');
        plt.show()
    def blocky_inv(self):
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

        print("\n -------------Running Blocky Simultaneous Inversion------------- \n")
        
        self.d_small = np.swapaxes(self.d_small, -1, 0)

        m_blocky, r_blocky = \
            pylops.avo.poststack.PoststackInversion(self.d_small, self.wav_est, m0=np.zeros_like(self.d_small), explicit=False, 
                                                    epsR=epsR_b, epsRL1=epsRL1_b,
                                                    **dict(mu=mu_b, niter_outer=niter_out_b, 
                                                        niter_inner=niter_in_b, show=True,
                                                        iter_lim=niter_b, damp=epsI_b))

        m_blocky = np.swapaxes(m_blocky, 0, -1)
        r_blocky = np.swapaxes(r_blocky, 0, -1)
        self.d_small = np.swapaxes(self.d_small, 0, -1)

        fig, ax = plt.subplots(1,1, figsize=(10, 5))
        c=ax.imshow(m_blocky[self.il_num - self.il_start, :, :].T, aspect='auto', cmap='seismic', vmin=-0.1*m_blocky.max(), vmax=0.1*m_blocky.max(), #<---------------------- Escolha a Inline substituindo zero por um indice valido
                    extent=[self.xl_start, self.xl_end, self.twt[int(self.t_max/self.dt)], self.twt[int(self.t_min/self.dt)]])
        plt.colorbar(c, ax=ax, pad=0.01)
        plt.grid(False)
        plt.show()
class Application(Funcs):
    def __init__(self) -> None:
        self.root = Tk()
        self.tela_load()
        self.frames()
        self.widgets_frame1()
        self.root.mainloop()
    def tela_load(self):
        self.root.title("Post-Stack Inversion GUI")
        self.root.configure(background='lightblue')
        self.root.geometry("600x400")
        self.root.resizable(True, True)
    def frames(self):
        self.frame_1 = Frame(self.root)
        self.frame_1.place(relx=0.02, rely=0.02, relwidth= 0.96, relheight= 0.5)   
    def widgets_frame1(self):
        
        self.bt_load1 = Button(self.frame_1, text="Browse file", command=self.load_file)
        self.bt_load1.place(relx=0.1, rely=0.1, relwidth= 0.2, relheight= 0.1)

        self.bt_load2 = Button(self.frame_1, text="Close", command=self.root.destroy)
        self.bt_load2.place(relx=0.1, rely=0.2, relwidth= 0.2, relheight= 0.1)
        
        self.file_label = Label(self.frame_1, text="")
        self.file_label.place(relx=0.0, rely=0.9, relwidth= 1, relheight= 0.1)
        
        self.il_number_label = Label(self.frame_1, text="Enter IL number for display")
        self.il_number_label.place(relx=0.35, rely=0.0, relwidth= 0.3, relheight= 0.1)
        
        self.il_entry = Entry(self.frame_1)
        self.il_entry.place(relx=0.4, rely=0.1, relwidth= 0.25, relheight= 0.1)
        
        self.il_entry_bt = Button(self.frame_1, text="Submit Il val", command=self.get_iline_num)
        self.il_entry_bt.place(relx=0.4, rely=0.2, relwidth= 0.2, relheight= 0.1)
        
        self.wav_bt = Button(self.frame_1, text="Run Inversion", command=self.blocky_inv)
        self.wav_bt.place(relx=0.8, rely=0.2, relwidth= 0.2, relheight= 0.1)

        self.parameter_button = Button(self.frame_1, text="Enter Parameters", command=self.open_parameter_window, state=DISABLED)
        self.parameter_button.place(relx=0.1, rely=0.3, relwidth= 0.2, relheight= 0.1)


        self.status_label = Label(self.frame_1, text="")
        self.status_label.place(relx=0.6, rely=0.1, relwidth= 0.2, relheight= 0.1)
    
Application()