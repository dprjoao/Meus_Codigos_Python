from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=2):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def Ricker_Wavelet(Peak_freq, Duration, Sampling) :
        t = np.linspace(-Duration,Duration,125)
        ricker = (1. -2.*(np.pi**2)*(Peak_freq**2)*(t**2))*np.exp(-(np.pi**2)*(Peak_freq**2)*(t**2))
        return t, ricker

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import freqz

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 125
    lowcut = 10
    highcut = 30

    # Plot the frequency response for a few different orders.
    '''plt.figure(1)
    plt.clf()
    for order in [3, 6, 9]:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        w, h = freqz(b, a, fs=fs, worN=2000)
        plt.plot(w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')'''

    # Filter a noisy signal.
    #T = 0.05
    #nsamples = T * fs
    #t = np.arange(0, nsamples) / fs
    #a = 0.02
    #f0 = 125
    #x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
    #x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
    #x += a * np.cos(2 * np.pi * f0 * t + .11)
    #x += 0.03 * np.cos(2 * np.pi * 2000 * t)
    #plt.figure(2)
    #plt.clf()
    #plt.plot(t, x, label='Noisy signal')

    
    Time , x = Ricker_Wavelet(250,0.5,0.004)

    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=1)
    plt.plot(Time, y, label='Filtered signal (%g Hz)' % fs)
    plt.xlabel('time (seconds)')
#plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')
    plt.show()

    ricker_fft = abs(np.fft.rfft(y))
    plt.plot(ricker_fft)
    plt.xlabel('Frequência (hz)')
    plt.ylabel('Amplitude relativa')
    plt.grid()
    plt.show()