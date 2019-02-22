import numpy as np
from scipy import signal
import time
import os
import matplotlib.pyplot as plt

fs = 500e6

b = [
    0.000044902640415897,
    0.000131577981915531,
    -0.000447118174500454,
    -0.000840741282312288,
    0.001577246966298278,
    0.003065661407864305,
    -0.003798350736721359,
    -0.008455018678000456,
    0.007209660430048435,
    0.019696600041232880,
    -0.011448893672724610,
    -0.041953084118684397,
    0.015680302330428341,
    0.091510017775400257,
    -0.018831478718331542,
    -0.313148038424453157,
    0.520013508464248986,
    -0.313148038424453157,
    -0.018831478718331542,
    0.091510017775400257,
    0.015680302330428341,
    -0.041953084118684397,
    -0.011448893672724610,
    0.019696600041232880,
    0.007209660430048435,
    -0.008455018678000456,
    -0.003798350736721359,
    0.003065661407864305,
    0.001577246966298278,
    -0.000840741282312288,
    -0.000447118174500454,
    0.000131577981915531,
    0.000044902640415897,
]

freq, H = signal.freqz(b)

filterFig, filterAxList = plt.subplots(2,1,sharex=True)
filterAxList[0].plot(1e-6*fs*freq/2/np.pi,20*np.log10(np.abs(H)),'b')
filterAxList[0].set_ylabel('Amplitude Response [dB]')
filterAxList[0].axis((0, 1e-6*fs/2, -150, 0))
filterAxList[0].grid()
filterAxList[1].plot(1e-6*fs*freq/2/np.pi,np.unwrap(np.angle(H)),'b')
filterAxList[1].set_xlabel('Frequency [MHz]')
filterAxList[1].set_ylabel('Phase Response [radians]')
filterAxList[1].grid()

chirpAmplitude = 0.3
chirpLength = 256
chirpSignal = chirpAmplitude*signal.hilbert(signal.chirp(np.arange(chirpLength)/fs, f0=150e6, f1=180e6, t1=(chirpLength-1)/fs, phi=90, method='linear')).astype(np.complex64)

intAmplitude = 1
intLength = 512
intSignal = intAmplitude*signal.hilbert(np.cos(2*np.pi*50e6*np.arange(intLength)/fs)).astype(np.complex64)

noiseVar = 0.01
group_delay = 16
rxSig_raw = np.concatenate((np.zeros(10), intSignal, np.zeros(64), chirpSignal, np.zeros(128)))
rxSig_noisy = rxSig_raw + np.sqrt(noiseVar/2)*(np.random.randn(len(rxSig_raw))+1j*np.random.randn(len(rxSig_raw)))
# rxSig_filtered = signal.lfilter(b,1,rxSig_noisy)[group_delay:]
rxSig_filtered = signal.filtfilt(b,1,rxSig_noisy)
import pdb; pdb.set_trace()

sigFig = plt.figure()
plt.plot(np.real(rxSig_raw),'r',label='Original')
plt.plot(np.real(rxSig_filtered),'b',label='Filtered')
plt.legend()
plt.grid()

freq_raw = np.fft.fftfreq(len(rxSig_raw), 1/fs)
freq_index = np.argsort(freq_raw)
freq_raw = freq_raw[freq_index]
fft_orig = np.fft.fft(rxSig_raw)[freq_index]
freq_filt = np.fft.fftfreq(len(rxSig_filtered), 1/fs)
freq_index = np.argsort(freq_filt)
freq_filt = freq_filt[freq_index]
fft_filtered = np.fft.fft(rxSig_filtered)[freq_index]

resFig = plt.figure()
plt.plot(1e-6*freq_raw, np.abs(fft_orig),'r',label='Original')
plt.plot(1e-6*freq_filt, np.abs(fft_filtered),'b',label='Filtered')
plt.xlabel('Frequency [MHz]')
plt.axis((-1e-6*fs/2, 1e-6*fs/2, 0, 40))
plt.grid()

MFraw = np.abs(np.convolve(rxSig_raw,chirpSignal[::-1].conjugate(),mode='same'))
MFfilt = np.abs(np.convolve(rxSig_filtered,chirpSignal[::-1].conjugate(),mode='same'))

mfFig = plt.figure()
plt.plot(MFraw,'r', label='Original')
plt.plot(np.argmax(MFraw)*np.ones(100), np.linspace(0, max(MFraw), 100),'r--')
plt.plot(MFfilt,'b', label='Filtered')
plt.plot(np.argmax(MFfilt)*np.ones(100), np.linspace(0, max(MFfilt), 100),'b--')
plt.legend()
plt.grid()

plt.show()