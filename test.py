imp = signal.unit_impulse(100, 'mid')
b, a = signal.butter(4, 0.2)
response = signal.lfilter(b, a, imp)
