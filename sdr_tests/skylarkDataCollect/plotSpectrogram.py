import numpy as np
from scipy import signal
import time
import os
import matplotlib.pyplot as plt
import pickle

fs = 10e6 # Sample rate in Hz
nperseg = int(2**17)
noverlap = int(2**17-2e3)

# Load data file
print("Loading Waveform Data")
# filename = 'recvData_20190204-162627_keep'
# filename = 'recvData_20190204-162730_keep'
# filename = 'recvData_20190204-162107_nothing'
filename = os.getcwd() + "/SavedData/" + 'recvData_20190205-165923_target.csv'
pulseFile = open(filename, 'r')
for line in pulseFile:
    input_interleaved = np.fromstring(line, dtype=np.float32, count=-1, sep=' ')
    sigLength = int(len(input_interleaved) / 2)
    interleavedValues = input_interleaved.reshape((sigLength, 2))
    realSamples = interleavedValues[:, 0]
    imagSamples = interleavedValues[:, 1]
    inputbuffer = realSamples + 1j * imagSamples
# recvData = pickle.load(open(os.getcwd() + "/skylarkDataCollect/SavedData" + filename + '.pkl', "rb"))
recvData = inputbuffer

# # Remove direct path
# print("Computing FFT")
# Fx = np.fft.fft(recvData)
# max_ind = np.argmax(np.abs(Fx))

# # Build interference matrix consisting of numModes carrier waveforms centered about the maximum
# print("Building Interference Matrix")
# numModes = 5
# H = np.zeros((len(recvData), 2*numModes+1), dtype=np.complex64)
# for ix, k in enumerate(np.arange(-numModes, numModes+1)):
# 	H[:,ix] = np.exp(1j*2*np.pi*(max_ind+k)*np.arange(len(recvData))/len(recvData))

# # Remove interference
# print("Removing Interference")
# x = np.linalg.solve(H.conj().T.dot(H), H.conj().T.dot(recvData))
# recvData2 = recvData - H.dot(x)

# Compute spectrogram
print("Computing Spectrogram")
freq, time, Sxx = signal.spectrogram(recvData, fs, noverlap=noverlap, nperseg=nperseg, return_onesided=False)
print(np.shape(Sxx))

# min_ind = np.argmin(np.abs(freq - (1e6-2e3)))
# max_ind = np.argmin(np.abs(freq - (1e6+2e3)))
min_ind = np.argmin(np.abs(freq - 1e6))
max_ind = np.argmin(np.abs(freq - 4e6))
freq = freq[min_ind:max_ind]
Sxx = Sxx[min_ind:max_ind,:]

# Mask dominant frequency
# window = 3
# mask = np.ones(np.shape(Sxx))
# ind = np.argmin(np.abs(freq-1e6))
# for ix in range(-window,window+1):
# 	mask[ind+ix,:] = 1e-20*np.ones(len(time))
# Sxx = np.multiply(mask,Sxx)

# Plot spectrogram
print("Plotting Spectrogram")
fig1 = plt.figure(figsize=(16, 8))
plt.pcolormesh(time, freq, np.log10(Sxx), 
	cmap='jet')
plt.axis((min(time), max(time), 1e6-2e3, 1e6+2e3))
plt.colorbar()

import pdb; pdb.set_trace()

fig1.savefig(filename + '.png')

# Target: [-10, -6] 
# Nothing: [-12, -7]

# # Load all data files to plot power vs. frequency
# print("Loading Waveform Data")
# filenames = ['recvData_20190204-162107_nothing', 'recvData_20190204-162627_keep', 'recvData_20190204-162730_keep'] 

# fftFig, fftAxList = plt.subplots(3,1,figsize=(16, 8),sharex=True)
# for ix, fn in enumerate(filenames):
# 	recvData = pickle.load(open(os.getcwd() + "/skylarkDataCollect/" + fn + '.pkl', "rb"))
# 	freq = np.fft.fftfreq(len(recvData), 1/fs)
# 	freq_index = np.argsort(freq)
# 	freq = freq[freq_index]
# 	Fx = np.fft.fft(recvData)[freq_index] 

# 	fftAxList[ix].plot(freq/1e6, 20*np.log10(np.abs(Fx)),'b')
# 	if ix == 2: fftAxList[ix].set_xlabel('Frequency (MHz)')
# 	fftAxList[ix].set_ylabel('Power (dB)')
# 	if ix == 0:
# 		fftAxList[ix].set_title('No Target Present')
# 	else:
# 		fftAxList[ix].set_title('Target Present')
# 	fftAxList[ix].set_xticks(1+np.arange(-25, 25)/2e3)
# 	fftAxList[ix].axis(((1e6-5e3)/1e6, (1e6+5e3)/1e6, 10, 100))
# 	fftAxList[ix].grid()
# fftFig.savefig('Power.png')

print("Done")