import segyio
import numpy as np
import matplotlib.pyplot as plt
import pylops
import tkinter as tk
from tkinter import filedialog

#-----------------------------------------------------------------------------------------------------
# Data
itmin = 600 # index of first time/depth sample in data used in colored inversion
itmax = 800 # number of samples of statistical wavelet

# Subsampling (can save file at the end only without subsampling)
jt = 1
jil = 1
jxl = 1

# Trace-by-Trace Inversion
epsI_tt = 1e-3 # damping

# Spatially simultaneous
niter_sr = 3 # number of iterations of lsqr
epsI_sr = 1e-4 # damping
epsR_sr = 1e2 # spatial regularization

# Blocky simultaneous
niter_out_b = 3 # number of outer loop iterations
niter_in_b = 1 # number of inner loop iterations
niter_b = 10 # number of iterations of lsqr
mu_b = 1e-1 # damping for data term
epsI_b = 1e-4 # damping
epsR_b = 0.1 # spatial regularization
epsRL1_b = 0.2 # blocky regularization

#-----------------------------------------------------------------------------------------------------
def load_file():
    filename = filedialog.askopenfilename()
    if filename:
        file_label.config(text=f"File loaded: {filename}")
        # Store the filename in a global variable or pass it to another function as needed
        global selected_file
        selected_file = filename

# Create the main window
root = tk.Tk()
root.title("File Loader")

# Create a button to load a file
load_button = tk.Button(root, text="Load File", command=load_file)
load_button.pack()

# Create a label to display the loaded file name
file_label = tk.Label(root, text="")
file_label.pack()

# Variable to store the selected file name
selected_file = ""

# Run the Tkinter event loop
root.mainloop()

#-----------------------------------------------------------------------------------------------------
#Loading seismic using Segyio lib

# string containing the path location of the seismic data at disk
segy_file_path = selected_file    #<-----------------Nome e caminho para a sismica

#loading stack using Segyio lib
stack = segyio.open(segy_file_path,iline=189, #<-----------------Byte da Inline
                xline=193) #<-----------------Byte da Cross

#Allocating IL, XL, Time axis in variables
il, xl, t = stack.ilines, stack.xlines, stack.samples

#Measuring Sample rate from data samples
dt = t[1] - t[0]

#Creating seismic cube format using segyio cube method
data_cube = segyio.cube(stack)
#Qc of the axis shapes
nil, nxl, nt = data_cube.shape

# Inlines information
il_start, il_end = il[0], il[-1]

# Crosslines information
xl_start, xl_end = xl[0], xl[-1]
#-----------------------------------------------------------------------------------------------------
#plotting data
fig, ax = plt.subplots(figsize=(10, 5))

c=ax.imshow(data_cube[1605-il_start, :, :].T, aspect='auto', cmap='gray_r', vmin=-3000, vmax=3000,
            extent=[xl_start, xl_end, t[-1], t[0]])

plt.colorbar(c, ax=ax, pad=0.01)
plt.grid(False)
ax.set_ylim(6000, 4500)
plt.show()
#-----------------------------------------------------------------------------------------------------
#Another wavelet estimation method

nt_wav = 16 # lenght of wavelet in samples
nfft = 2**8 # lenght of fft

# time axis for wavelet
t_wav = np.arange(nt_wav) * (dt/1000) 
t_wav = np.concatenate((np.flipud(-t_wav[1:]), t_wav), axis=0)

# estimate wavelet spectrum
wav_est_fft = np.mean(np.abs(np.fft.fft(data_cube[..., int(4600/dt):int(5600/dt)], nfft, axis=-1)), axis=(0, 1))#<-----------------Janela da FFT em indice
fwest = np.fft.fftfreq(nfft, d=dt/1000)

# create wavelet in time
wav_est = np.real(np.fft.ifft(wav_est_fft)[:nt_wav])
wav_est = np.concatenate((np.flipud(wav_est[1:]), wav_est), axis=0)
wav_est = wav_est / wav_est.max()
wcenter = np.argmax(np.abs(wav_est))

# display wavelet
fig, axs = plt.subplots(1, 2, figsize=(20, 5))
fig.suptitle('Statistical wavelet estimate')
axs[0].plot(fwest[:nfft//2], wav_est_fft[:nfft//2], 'k')
axs[0].set_title('Frequency')
axs[1].plot(wav_est, 'k')
axs[1].set_title('Time');
plt.show()
#-----------------------------------------------------------------------------------------------------
# swap time axis to first dimension
d_small = data_cube[..., int(4600/dt):int(5600/dt)]
d_small = np.swapaxes(d_small, -1, 0)

print("\n -------------Running Post-Stack Inversion------------- \n")

m_blocky, r_blocky = \
    pylops.avo.poststack.PoststackInversion(d_small, wav_est, m0=np.zeros_like(d_small), explicit=False, 
                                            epsR=epsR_b*10, epsRL1=epsRL1_b*10,
                                            **dict(mu=mu_b, niter_outer=niter_out_b, 
                                                   niter_inner=niter_in_b, show=True,
                                                   iter_lim=niter_b, damp=epsI_b))

m_blocky = np.swapaxes(m_blocky, 0, -1)
r_blocky = np.swapaxes(r_blocky, 0, -1)

fig, ax = plt.subplots(1,1, figsize=(10, 5))
c=ax.imshow(m_blocky[1605 - il_start, :, :].T, aspect='auto', cmap='seismic', vmin=-0.1*m_blocky.max(), vmax=0.1*m_blocky.max(), #<---------------------- Escolha a Inline substituindo zero por um indice valido
            extent=[xl_start, xl_end, t[int(5600/dt)], t[int(4600/dt)]])
plt.colorbar(c, ax=ax, pad=0.01)
plt.grid(False)
plt.show()
#-----------------------------------------------------------------------------------------------------