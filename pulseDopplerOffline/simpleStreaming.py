#!/usr/bin/python3

import numpy as np
import sys
import SoapySDR
import time
from SoapySDR import * #SOAPY_SDR_constants

freq = 2.55e9
rate = 5e6
rxGain = 10
txGain = 40
delay = int(10e6)
nsamps=8092

txSerial = "RF3E000075"
txChan = 0
rxChan = 1

sdr = SoapySDR.Device(dict(driver="iris", serial = serial))

# Init Rx
sdr.setSampleRate(SOAPY_SDR_RX, 0, rate)
sdr.setFrequency(SOAPY_SDR_RX, rxChan, "RF", freq)
sdr.setGain(SOAPY_SDR_RX, rxChan, rxGain)
sdr.setAntenna(SOAPY_SDR_RX, rxChan, "TRX")
sdr.setFrequency(SOAPY_SDR_RX, rxChan, "BB", 0) #don't use cordic
sdr.setDCOffsetMode(SOAPY_SDR_RX, rxChan, False) #dc removal on rx #we'll remove this in post-processing

# Init Tx
sdr.setSampleRate(SOAPY_SDR_TX, txChan, rate)
sdr.setFrequency(SOAPY_SDR_TX, txChan, "RF", freq)
sdr.setGain(SOAPY_SDR_TX, txChan, txGain)
sdr.setAntenna(SOAPY_SDR_TX, txChan, "TRX")

#Init Buffers
sampsRecv = np.zeros(nsamps, dtype=np.complex64)
sampsSend = np.zeros(nsamps, dtype=np.complex64)

#Init Streams
rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0], {})
txStream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0], {})

ts = sdr.getHardwareTime() + delay
sdr.activateStream(txStream, SOAPY_SDR_HAS_TIME, ts)
sdr.activateStream(rxStream, SOAPY_SDR_HAS_TIME, ts)
	
t=time.time()
total_samps = 0
while True:
	sr = sdr.writeStream(txStream, [sampsSend], nsamps)
	if sr.ret != nsamps:
		print("Bad Write!!!")
	#do some tx processing here
		
	sr = sdr.readStream(rxStream, [sampsRecv], nsamps, timeoutUs=int(10e6))		
	if sr.ret != nsamps:
		print("Bad Read!!!")
	#do some rx processing here
	
	total_samps += nsamps
	if time.time() - t > 1:
		t=time.time()
		print("Total Samples Sent and Received: %i" % total_samps)
	
	#It is probably good to sleep here, but the readStream will block sufficiently
	#it just depends on your processing

