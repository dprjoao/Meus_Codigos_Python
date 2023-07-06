#Programa que implementa computacionalmente a wavelet de Ricker

import sys
import numpy as np 
import matplotlib.pyplot as plt # type: ignore
from numpy.typing import NDArray
from typing import Tuple, overload

def Ricker_Wavelet(Peak_freq: float, Samples: float, Dt: float) -> Tuple[NDArray, NDArray]:
    t = np.arange(Samples)*(Dt/1000)
    t = np.concatenate((np.flipud(-t[1:]), t), axis=0)
    ricker = (1. -2.*(np.pi**2)*(Peak_freq**2)*(t**2))*np.exp(-(np.pi**2)*(Peak_freq**2)*(t**2))
    return t, ricker

Peak_freq = float(sys.argv[1])
Samples = float(sys.argv[2])
Sampling = float(sys.argv[3])

Time, Ricker_wl = Ricker_Wavelet(Peak_freq, Samples, Sampling)

#Ricker wavelet spectrum
freqs = np.fft.rfftfreq(Time.shape[0], d=Sampling)*1000
a = np.fft.rfft(Ricker_wl)
A = np.abs(a)

# display wavelet
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('Ricker Wavelet')
axs[0].plot(freqs, A, 'k')
axs[0].set_title('Frequency')
axs[1].plot(Time, Ricker_wl, 'k')
axs[1].set_title('Time')
plt.show()