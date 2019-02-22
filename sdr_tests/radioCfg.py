
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
serials = ["RF3E000075","RF3E000069"]
txGain = 30 
rxGain = 30
rate = 10e6
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