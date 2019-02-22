########################################################################
## Generate Single Mode Sinusoidal Signal
########################################################################

import SoapySDR
from SoapySDR import * #SOAPY_SDR_ constants
import numpy as np
from scipy import signal
from optparse import OptionParser
import time
import os
import matplotlib.pyplot as plt

# serial = "31119E8" # USRP
serial = "RF3E000075" # Skylark
txChan = 0
txGain = 70
amplitude = 1
rate = 18e6
carrierFreq = 2575e6 # 150 MHz carrier
# carrierFreq = 5800e6 # 5.8 GHz carrier
# txAnt = "TX/RX" # USRP
txAnt = "TRX"

# Define time vector
NumTxSamps = 4096
t = np.arange(NumTxSamps)/rate

# Generate signal
# txPulse = np.exp(1j*2*np.pi*carrierFreq*t).astype(np.complex64)
txPulse = np.ones(NumTxSamps, np.complex64)

# Setup device
# tx_sdr = SoapySDR.Device(dict(driver="uhd", serial = serial))
tx_sdr = SoapySDR.Device(dict(driver="iris", serial = serial))
if not tx_sdr.hasHardwareTime():
    raise Exception('this device does not support timed streaming')

#set sample rate
tx_sdr.setSampleRate(SOAPY_SDR_TX, txChan, rate)
print("Actual Tx Rate %f Msps"%(tx_sdr.getSampleRate(SOAPY_SDR_TX, txChan)/1e6))

#set antenna
tx_sdr.setAntenna(SOAPY_SDR_TX, txChan, txAnt)

#set overall gain
tx_sdr.setGain(SOAPY_SDR_TX, txChan, txGain)
print("Actual Tx Gain %f dB"%(tx_sdr.getGain(SOAPY_SDR_TX, txChan)))

#set front-end frequency
tx_sdr.setFrequency(SOAPY_SDR_TX, txChan, carrierFreq)
# tx_sdr.setFrequency(SOAPY_SDR_TX, txChan, 0)


def continuousTx(): 

	#create tx streams
    print("Create Tx streams")
    txStream = tx_sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [txChan])

    flag = True
    while flag:
    	#transmit stream
    	tx_sdr.activateStream(txStream)
    	txTime0 = tx_sdr.getHardwareTime() + long(0.1e9) #100ms
    	# txPulse = np.exp(1j*2*np.pi*carrierFreq*t).astype(np.complex64)
    	# txPulse = np.ones(NumTxSamps, np.complex64)
    	sr = tx_sdr.writeStream(txStream, [txPulse], len(txPulse))
    	if sr.ret != len(txPulse): raise Exception('transmit failed %s'%str(sr))
    	# txFlags = SOAPY_SDR_HAS_TIME | SOAPY_SDR_END_BURST
    	# sr = tx_sdr.writeStream(txStream, [txPulse], len(txPulse), txFlags, txTime0)
    	# if sr.ret != len(txPulse): raise Exception('transmit failed %s'%str(sr))