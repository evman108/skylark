
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

import SoapySDR
from SoapySDR import * #SOAPY_SDR_ constants
import numpy as np
from optparse import OptionParser
import time
import os
import math
import sys
import matplotlib.pyplot as plt

def cfloat2uint32(arr, order='IQ'):
		arr_i = (np.real(arr) * 32767).astype(np.uint16)
		arr_q = (np.imag(arr) * 32767).astype(np.uint16)
		if order == 'IQ':
			return np.bitwise_or(arr_q ,np.left_shift(arr_i.astype(np.uint32), 16))
		else:
			return np.bitwise_or(arr_i ,np.left_shift(arr_q.astype(np.uint32), 16))
	
def uint32tocfloat(arr, order='IQ'):
	arr_hi = ((np.right_shift(arr, 16).astype(np.int16))/32768.0)
	arr_lo = (np.bitwise_and(arr, 0xFFFF).astype(np.int16))/32768.0
	if order == 'IQ':
		return (arr_hi + 1j*arr_lo).astype(np.complex64)
	else:
		return (arr_lo + 1j*arr_hi).astype(np.complex64)

num_pulses = 16
serials = ["RF3E000075"]
txGain = 30 
rxGain = 30
rate = 30e6
freq = 2575e6
txAnt = "TRX"
rxAnt = "TRX"



num_sdrs=len(serials)
num_samps=1024*num_sdrs*4
num_plots = (num_sdrs-1)*2
print(num_sdrs)

# Read transmit buffer from stdin
pulseFile = open("skylark_transmit.txt", 'r')
for line in pulseFile:
	input_interleaved = np.fromstring(line, dtype=np.float32, count=-1, sep=' ')
	sigLength = int(len(input_interleaved) / 2)
	interleavedValues = input_interleaved.reshape((sigLength, 2))
	realSamples = interleavedValues[:, 0]
	imagSamples = interleavedValues[:, 1]
	inputbuffer = realSamples + 1j * imagSamples

num_samps = num_pulses*len(inputbuffer)
print(num_samps)

from pulseDopplerTxRx import MIMO_SDR

sdrs = [SoapySDR.Device(dict(driver="iris", serial = serial)) for serial in serials]
for serial in serials:
	print(serial)
#	SoapySDR.Device(dict(driver="iris", serial = serial))

tx_sdrs = sdrs[0:len(sdrs)//2]
rx_sdrs = sdrs[len(sdrs)//2:]
trig_sdr = sdrs[0]


print("Using %i tx Irises and %i rx Irises." % (len(tx_sdrs), len(rx_sdrs)) )

#override default settings
for sdr in sdrs:
	for chan in [0, 1]:
		sdr.setSampleRate(SOAPY_SDR_RX, chan, rate)
		# sdr.setBandwidth(SOAPY_SDR_RX, chan, bw)
		sdr.setGain(SOAPY_SDR_RX, chan, rxGain)
		sdr.setFrequency(SOAPY_SDR_RX, chan, "RF", freq)
		sdr.setAntenna(SOAPY_SDR_RX, chan, rxAnt)
		sdr.setFrequency(SOAPY_SDR_RX, chan, "BB", 0) #don't use cordic
		sdr.setDCOffsetMode(SOAPY_SDR_RX, chan, True) #dc removal on rx

		sdr.setSampleRate(SOAPY_SDR_TX, chan, rate)
		# sdr.setBandwidth(SOAPY_SDR_TX, chan, bw)
		sdr.setGain(SOAPY_SDR_TX, chan, txGain)
		sdr.setFrequency(SOAPY_SDR_TX, chan, "RF", freq)
		sdr.setAntenna(SOAPY_SDR_TX, chan, txAnt)
		sdr.setFrequency(SOAPY_SDR_TX, chan, "BB", 0) #don't use cordic

		# NO DOCUMENTATION ON THESE SETTINGS
		sdr.writeSetting(SOAPY_SDR_RX, chan, 'CALIBRATE', 'SKLK')
		sdr.writeSetting(SOAPY_SDR_TX, chan, 'CALIBRATE', 'SKLK')
		sdr.writeSetting('SPI_TDD_MODE', 'MIMO')

# NO DOCUMENTATION ON THESE SETTINGS
trig_sdr.writeSetting('SYNC_DELAYS', "")
for sdr in sdrs: sdr.setHardwareTime(0, "TRIGGER")


#create rx streams
rxStreams = [sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0, 1], {"remote:prot":"tcp", "remote:mtu":"1024"}) for sdr in rx_sdrs]
num_rx_r = len(rx_sdrs)*2
sampsRecv = [np.empty(num_samps).astype(np.complex64) for r in range(num_rx_r)]
print("Receiving chunks of %i" % len(sampsRecv[0]))

#create tx stream
txStreams = []
for sdr in tx_sdrs:
	txStream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0], {"REPLAY": 'true'})
	sdr.activateStream(txStream)
	txStreams.append(txStream)


#create our own sinusoid
if inputbuffer is None:
	Ts = 1/rate
	s_length = 768*4
	s_freq = 1e6
	s_time_vals = np.array(np.arange(0,s_length)).transpose()*Ts
	s = np.exp(s_time_vals*1j*2*np.pi*s_freq).astype(np.complex64)*.25
	num_tx_r = len(tx_sdrs)*2
	sampsToSend = [np.zeros(int(num_samps/4)).astype(np.complex64) for r in range(num_tx_r)]
else:
	s = inputbuffer
	s_length = len(s)
	num_tx_r = len(tx_sdrs)*2
	print(int(num_samps/4))
	sampsToSend = [np.zeros(int(num_samps/num_pulses)).astype(np.complex64) for r in range(num_tx_r)]

#samples to send is a two channel array of complex floats
for r in range(num_tx_r):
	print('R-',r)
	sampsToSend[r] = s
	print(sampsToSend)

print("Done initializing")


#clear out socket buffer from old requests
for r,sdr in enumerate(rx_sdrs):
	rxStream = rxStreams[r]
	sr = sdr.readStream(rxStream, [sampsRecv[r*2][:], sampsRecv[r*2+1][:]], len(sampsRecv[0]), timeoutUs = 0)
	while sr.ret != SOAPY_SDR_TIMEOUT:
		sr = sdr.readStream(rxStream, [sampsRecv[r*2][:], sampsRecv[r*2+1][:]], len(sampsRecv[0]), timeoutUs = 0)

tx_sdrs[0].writeRegisters('TX_RAM_A', 0, cfloat2uint32(inputbuffer).tolist())

time.sleep(0.1)

#receive a waveform at the same time
for r,sdr in enumerate(rx_sdrs):
	rxStream = rxStreams[r]
	flags = SOAPY_SDR_WAIT_TRIGGER | SOAPY_SDR_END_BURST
	#flags = SOAPY_SDR_HAS_TIME | SOAPY_SDR_END_BURST
	sdr.activateStream(rxStream, flags, 0, len(sampsRecv[0]))

trig_sdr.writeSetting("TRIGGER_GEN", "")
tx_sdrs[0].writeSetting("TX_REPLAY", str(len(inputbuffer)))

#time.sleep(.5)

for r,sdr in enumerate(rx_sdrs):
	
	rxStream = rxStreams[r]
	sr = sdr.readStream(rxStream, [sampsRecv[r*2], sampsRecv[r*2+1]], len(sampsRecv[0]), timeoutUs=int(1e6))
	if sr.ret != len(sampsRecv[0]):
		print("Bad read!!!")
		
	#remove residual DC offset
	sampsRecv[r*2][:] -= np.mean(sampsRecv[r*2][:])
	sampsRecv[r*2+1][:] -= np.mean(sampsRecv[r*2+1][:])

#look at any async messages
print('Issues:')
for r,sdr in enumerate(tx_sdrs):
	txStream = txStreams[r]
	sr = sdr.readStreamStatus(txStream, timeoutUs=int(1e6))
	print(sr)

#cleanup streams
print("Cleanup streams")
for r,sdr in enumerate(tx_sdrs):
	sdr.deactivateStream(txStreams[r])
	sdr.closeStream(txStreams[r])
#for sdr,rxStream in (rx_sdrs,rxStreams):
for r,sdr in enumerate(rx_sdrs):
	sdr.deactivateStream(rxStreams[r])
	sdr.closeStream(rxStreams[r])
print("Done!")



t = np.arange(inputbuffer.size)
waveFormFig, waveformAxList = plt.subplots(3,1)
waveformAxList[0].plot(t,np.real(sampsToSend[0]))
waveformAxList[0].plot(t,np.imag(sampsToSend[0]))
t = np.arange(sampsRecv[0].size)
waveformAxList[1].plot(t,np.real(sampsRecv[0]))
waveformAxList[1].plot(t,np.imag(sampsRecv[0]))
waveformAxList[2].plot(t,np.real(sampsRecv[1]))
waveformAxList[2].plot(t,np.imag(sampsRecv[1]))

plt.show()