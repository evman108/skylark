#!/usr/bin/python3

import numpy as np
import sys
import SoapySDR
import time
from SoapySDR import * #SOAPY_SDR_constants
import matplotlib.pyplot as plt

freq = 2.60e9
rate = 5e6
rxGain = 30
txGain = 50
delay = int(1e6)
streamPktSize = 473
nsampsToBuffer=  17*streamPktSize 
writeTime = nsampsToBuffer/rate
totalNumSamps = nsampsToBuffer*4
reportInterval = 1
overTheAir = False # else simulate
numPktsToPreBufferOnTx = 2

txSerial = "RF3E000075" # head of the chain
rxSerial = "RF3E000069"
txChan = 0
rxChan = 0


#Design the transmit signal
Ts = 1/rate
s_freq = 1e4
s_time_vals = np.array(np.arange(0,totalNumSamps)).transpose()*Ts

txVec = np.exp(s_time_vals*1j*2*np.pi*s_freq).astype(np.complex64)*.5

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
	tx_flags = SOAPY_SDR_HAS_TIME;
	tx_flags = 0;
	tx_sdr.activateStream(txStream, tx_flags, ts)
	timeOfTxStreamActivation = time.time()

	rx_flags = SOAPY_SDR_HAS_TIME;
	rx_sdr.activateStream(rxStream, rx_flags, ts)
	timeOfRxStreamActivation = time.time()

	
#timeOfLastPrintout=time.time()
print("Commencing TX/RX")
total_samps = 0
rxBuffs = np.array([], np.complex64)
txVecSent = np.array([], np.complex64)

txSampsSent = 0;
rxSampsSent = 0;
rxBuffLag = 0;

# go ahead an load a handful of 
# pkts in the tx buffer before
# we start iterative polling. 
# This is to avoid underflows on tx. 
for ix in range(numPktsToPreBufferOnTx):
	sampsSend = txVec[txSampsSent:txSampsSent+nsampsToBuffer]

	if overTheAir:
		sr = tx_sdr.writeStream(txStream, [sampsSend], nsampsToBuffer)
		if sr.ret != nsampsToBuffer:
			raise Exception('tx fail - Bad write')
		txVecSent = np.concatenate((txVec, sampsSend[:sr.ret]))
	else:
		txVecSent = np.concatenate((txVec, sampsSend))
		rxLag = nsampsToBuffer*(numPktsToPreBufferOnTx+1)
	txSampsSent += nsampsToBuffer
	rxBuffLag += nsampsToBuffer

sampsRecv = np.zeros(nsampsToBuffer).astype(np.complex64)
while (txSampsSent < totalNumSamps) | (rxSampsSent < totalNumSamps):

	loopStart = time.time()

	if txSampsSent < totalNumSamps:
		print("tx")
		sampsSend = txVec[txSampsSent:txSampsSent+nsampsToBuffer]
		txSampsSent += nsampsToBuffer
	else:
		sampsSend = np.zeros(nsampsToBuffer).astype(np.complex64)
		rxBuffLag -= nsampsToBuffer


	if overTheAir:
		
		sr = tx_sdr.writeStream(txStream, [sampsSend], nsampsToBuffer)
		if sr.ret != nsampsToBuffer:
			raise Exception('tx fail - Bad write')
	
		txVecSent = np.concatenate((txVec, sampsSend[:sr.ret]))

		sr = rx_sdr.readStream(rxStream, [sampsRecv], nsampsToBuffer, timeoutUs=int(10e6))		
		if sr.ret != nsampsToBuffer:
			raise Exception('receive fail - Bad Read')
	
		rxBuffs = np.concatenate((rxBuffs, sampsRecv[:sr.ret]))

	else:
		
		sampsRecv = txVec[rxSampsSent: rxSampsSent+nsampsToBuffer]
		rxBuffs = np.concatenate((rxBuffs, sampsRecv))
	
	rxSampsSent += nsampsToBuffer
	print("Time in loop: %f:" % (time.time() - loopStart))


#cleanup streams
if overTheAir:
	tx_sdr.deactivateStream(txStream)
	tx_sdr.closeStream(txStream)
	rx_sdr.deactivateStream(rxStream)
	rx_sdr.closeStream(rxStream)


	
	#It is probably good to sleep here, but the readStream will block sufficiently
	#it just depends on your processing

print("\nDone reading and writing")
waveFormFig, waveFormAxList = plt.subplots(2,1,sharex=True)
waveFormAxList[0].plot(np.real(txVec),'b')
waveFormAxList[0].plot(np.imag(txVec),'r')
waveFormAxList[0].set_title('Tx')
waveFormAxList[1].plot(np.real(rxBuffs),'b')
waveFormAxList[1].plot(np.imag(rxBuffs),'r')
waveFormAxList[1].set_title('Rx')

print("\nDone!")

plt.show()