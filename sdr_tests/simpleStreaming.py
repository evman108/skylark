#!/usr/bin/python3

import numpy as np
import sys
import SoapySDR
import time
from SoapySDR import * #SOAPY_SDR_constants

if __name__ == '__main__':
	freq = 2.45e9
	rate = 5e6
	rxGain = 10
	txGain = 40
	delay = int(10e6)
	nsamps=8092
	
	sdr = SoapySDR.Device(dict(driver="iris", serial = sys.argv[1]))

	# Init Rx
	sdr.setSampleRate(SOAPY_SDR_RX, 0, rate)
	sdr.setFrequency(SOAPY_SDR_RX, 0, "RF", freq)
	sdr.setGain(SOAPY_SDR_RX, 0, rxGain)
	sdr.setAntenna(SOAPY_SDR_RX, 0, "RX")
	sdr.setFrequency(SOAPY_SDR_RX, 0, "BB", 0) #don't use cordic
	sdr.setDCOffsetMode(SOAPY_SDR_RX, 0, False) #dc removal on rx #we'll remove this in post-processing
	
	# Init Tx
	sdr.setSampleRate(SOAPY_SDR_TX, 0, rate)
	sdr.setFrequency(SOAPY_SDR_TX, 0, "RF", freq)
	sdr.setGain(SOAPY_SDR_TX, 0, txGain)
	sdr.setAntenna(SOAPY_SDR_TX, 0, "TRX")
	
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

