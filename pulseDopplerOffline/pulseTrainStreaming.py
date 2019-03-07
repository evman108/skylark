#!/usr/bin/python3
overTheAir = False # else simulate
freq = 2.60e9
rate = 5e6
rxGain = 25
txGain = 50
delay = int(1e6)
streamPktSize = 473
nsampsToBuffer=  17*streamPktSize 
writeTime = nsampsToBuffer/rate
numBufferWritesPerDwell = 30
totalNumSamps = nsampsToBuffer * numBufferWritesPerDwell
reportInterval = 1

numPktsToPreBufferOnTx = 2
extraZerosToAppend = 17
dwellTime = totalNumSamps/rate

txSerial = "RF3E000075" # head of the chain
rxSerial = "RF3E000069"
txChan = 0
rxChan = 0


import numpy as np
import scipy as sp
import sys
import time
import matplotlib.pyplot as plt

if overTheAir:
	import SoapySDR
	from SoapySDR import * #SOAPY_SDR_constants
else:
	# setup noise params
	numTargets = 1
	snr = 1e7 # linear
	maxDelay = 10 # samples
	maxDopper = 10e3 # Hz 


#Design the transmit signal
header_len = 140
header = np.zeros(header_len).astype(np.complex64)


pulseFile = open("skylark_transmit2.txt", 'r')
for line in pulseFile:
    input_interleaved = np.fromstring(line, dtype=np.float32, count=-1, sep=' ')
    sigLength = int(len(input_interleaved) / 2)
    interleavedValues = input_interleaved.reshape((sigLength, 2))
    realSamples = interleavedValues[:, 0]
    imagSamples = interleavedValues[:, 1]
    inputbuffer = realSamples + 1j * imagSamples
txProtoPulse = np.concatenate((inputbuffer, np.zeros(extraZerosToAppend).astype(np.complex64)))
pulseLength = len(txProtoPulse)

# numPulses = 1*1024
numPulses = int((totalNumSamps-header_len)/pulseLength)
pri = pulseLength/rate
print("Transmitting %d pulses over a %.1f ms dwell" % (numPulses,dwellTime*1e3))
footer_len = (totalNumSamps-header_len)%pulseLength
footer = np.zeros(footer_len).astype(np.complex64)
txVec = header
for ix in range(numPulses):
    txVec = np.concatenate((txVec,txProtoPulse))
txVec = np.concatenate((txVec,footer))

numTxSamps = len(txVec)
numRxSamps = numTxSamps+100

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

numRxSamps = len(rxBuffs);
assert(numRxSamps == numTxSamps)

#cleanup streams
if overTheAir:
	tx_sdr.deactivateStream(txStream)
	tx_sdr.closeStream(txStream)
	rx_sdr.deactivateStream(rxStream)
	rx_sdr.closeStream(rxStream)
else:
	# apply noise, delay, and doppler
	delay = maxDelay;
	# apply delay
	rxBuffs = np.roll(rxBuffs,delay)
	#apply Doppler
	dopplerShift =  1/(10*pulseLength)
	dopplerModulation= np.exp(2*1j*np.pi*dopplerShift*np.arange(numRxSamps)) 
	rxBuffs = dopplerModulation*rxBuffs
	#apple noise 
	noiseVec = 1/np.sqrt(snr)*(np.random.randn(numRxSamps) + 1j*np.random.randn(numRxSamps))
	rxBuffs = rxBuffs + noiseVec
	
	#It is probably good to sleep here, but the readStream will block sufficiently
	#it just depends on your processing

# Range-Doppler Processing
# strip off the header
txVec = txVec[header_len:]
rxBuffs = rxBuffs[header_len:]


pulseCompressedTx = np.convolve(txProtoPulse[::-1].conjugate(),txVec)
pulseCompressedRx = np.convolve(txProtoPulse[::-1].conjugate(),rxBuffs)

## sort data into slow time and fast time

pulseCompressionPeak = np.max(abs(np.convolve(txProtoPulse[::-1], txProtoPulse.conjugate()))) 
zeroRangeBins = sp.argwhere(pulseCompressedTx==pulseCompressionPeak)
assert(zeroRangeBins[1][0] - zeroRangeBins[0][0] == pulseLength)
firstZeroRangeBin = zeroRangeBins[0][0]
lastZeroRangeBin = zeroRangeBins[-1][0]

rangeSamplesTx = np.reshape(pulseCompressedTx[firstZeroRangeBin:lastZeroRangeBin+pulseLength],
	[numPulses,pulseLength])

rangeSamplesRx = np.reshape(pulseCompressedRx[firstZeroRangeBin:lastZeroRangeBin+pulseLength],
	[numPulses,pulseLength])

## TO DO: compute fft over slow time to realize the RD map

print("\nDone reading and writing")
waveFormFig, waveFormAxList = plt.subplots(2,1,sharex=True)
waveFormAxList[0].plot(np.real(txVec),'b')
waveFormAxList[0].plot(np.imag(txVec),'r')
waveFormAxList[0].set_title('Tx')
waveFormAxList[1].plot(np.real(rxBuffs),'b')
waveFormAxList[1].plot(np.imag(rxBuffs),'r')
waveFormAxList[1].set_title('Rx')

matchedFilterFig, matchedFilterAxList = plt.subplots(2,1,sharex=True)
matchedFilterAxList[0].plot(np.real(pulseCompressedTx),'b')
matchedFilterAxList[0].plot(np.imag(pulseCompressedTx),'r')
matchedFilterAxList[0].set_title('Tx Compressed')
matchedFilterAxList[1].plot(np.real(pulseCompressedRx),'b')
matchedFilterAxList[1].plot(np.imag(pulseCompressedRx),'r')
matchedFilterAxList[1].set_title('Rx Compressed')

print("\nDone!")

plt.show()