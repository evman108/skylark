#!/usr/bin/python3

import numpy as np
import sys
import SoapySDR
import time
from SoapySDR import * #SOAPY_SDR_constants
import matplotlib.pyplot as plt

freq = 2.60e9
rate = 5e6
rxGain = 20
txGain = 50
delay = int(1e6)
streamPktSize = 473
nsamps=  17*streamPktSize 
writeTime = nsamps/rate
totalNumSamps = nsamps*50
reportInterval = 1
overTheAir = False # else simulate

txSerial = "RF3E000075" # head of the chain
rxSerial = "RF3E000069"
txChan = 0
rxChan = 0


#Design the transmit signal
Ts = 1/rate
s_freq = 1e4
s_time_vals = np.array(np.arange(0,nsamps)).transpose()*Ts
sampsSend = np.exp(s_time_vals*1j*2*np.pi*s_freq).astype(np.complex64)*.5
sampsRecv = np.zeros(nsamps).astype(np.complex64)


if overTheAir: # setup the radios

	tx_sdr = SoapySDR.Device(dict(driver="iris", serial = txSerial))
	rx_sdr = SoapySDR.Device(dict(driver="iris", serial = rxSerial))

	# Init Rx
	rx_sdr.setSampleRate(SOAPY_SDR_RX, rxChan, rate)
	rx_sdr.setFrequency(SOAPY_SDR_RX, rxChan, "RF", freq)
	rx_sdr.setGain(SOAPY_SDR_RX, rxChan, rxGain)
	rx_sdr.setAntenna(SOAPY_SDR_RX, rxChan, "TRX")
	rx_sdr.setFrequency(SOAPY_SDR_RX, rxChan, "BB", 0) #don't use cordic
	rx_sdr.setDCOffsetMode(SOAPY_SDR_RX, rxChan, False) #dc removal on rx #we'll remove this in post-processing

	# Init Tx
	tx_sdr.setSampleRate(SOAPY_SDR_TX, txChan, rate)
	tx_sdr.setFrequency(SOAPY_SDR_TX, txChan, "RF", freq)
	tx_sdr.setGain(SOAPY_SDR_TX, txChan, txGain)
	tx_sdr.setAntenna(SOAPY_SDR_TX, txChan, "TRX")

	tx_sdr.writeSetting('SYNC_DELAYS', "")
	for sdr in [tx_sdr, rx_sdr]: sdr.setHardwareTime(0, "TRIGGER")
	tx_sdr.writeSetting("TRIGGER_GEN", "")


	#Init Streams
	rxStream = rx_sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0], {})
	txStream = tx_sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0], {})

	ts = tx_sdr.getHardwareTime() + delay
	#tx_flags = SOAPY_SDR_HAS_TIME;
	tx_flags = 0;
	tx_sdr.activateStream(txStream, tx_flags, ts)
	timeOfTxStreamActivation = time.time()

	rx_flags= 0;
	rx_sdr.activateStream(rxStream, rx_flags, ts)
	timeOfRxStreamActivation = time.time()
	
#timeOfLastPrintout=time.time()
print("Commencing TX/RX")
total_samps = 0
rxBuffs = np.array([], np.complex64)
txVec = np.array([], np.complex64)

if overTheAir:
	sr = tx_sdr.writeStream(txStream, [sampsSend], nsamps)
	if sr.ret != nsamps:
		raise Exception('tx fail - Bad write')
	txVec = np.concatenate((txVec, sampsSend[:sr.ret]))
else:
	txVec = np.concatenate((txVec, sampsSend))

while total_samps < totalNumSamps:

	loopStart = time.time()

	if overTheAir:
	
		sr = tx_sdr.writeStream(txStream, [sampsSend], nsamps)
		if sr.ret != nsamps:
			raise Exception('tx fail - Bad write')
	
		#do some tx processing here
		txVec = np.concatenate((txVec, sampsSend[:sr.ret]))

		sr = rx_sdr.readStream(rxStream, [sampsRecv], nsamps, timeoutUs=int(10e6))		
		if sr.ret != nsamps:
			raise Exception('receive fail - Bad Read')
	
		#do some rx processing here
		rxBuffs = np.concatenate((rxBuffs, sampsRecv[:sr.ret]))

		print("Time in loop: %f:" % (time.time() - loopStart))

	else:

		txVec = np.concatenate((txVec, sampsSend))
		sampsRecv = sampsSend
		rxBuffs = np.concatenate((rxBuffs, sampsRecv))
	
	total_samps += nsamps


	
	#It is probably good to sleep here, but the readStream will block sufficiently
	#it just depends on your processing


waveFormFig, waveFormAxList = plt.subplots(2,1,sharex=True)
waveFormAxList[0].plot(np.real(txVec),'b')
waveFormAxList[0].plot(np.imag(txVec),'r')
#waveFormAxList[0].set_title('Desired Transmit')
waveFormAxList[1].plot(np.real(rxBuffs),'b')
waveFormAxList[1].plot(np.imag(rxBuffs),'r')
#[1].set_title('Received Data')


#cleanup streams
if overTheAir:
	print("\nCleanup streams")
	tx_sdr.deactivateStream(txStream)
	rx_sdr.deactivateStream(rxStream)
	rx_sdr.closeStream(rxStream)
	tx_sdr.closeStream(txStream)

print("Done!")

plt.show()