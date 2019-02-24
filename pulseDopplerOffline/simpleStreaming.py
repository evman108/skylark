#!/usr/bin/python3

import numpy as np
import sys
import SoapySDR
import time
from SoapySDR import * #SOAPY_SDR_constants
import matplotlib.pyplot as plt

freq = 2.50e9
rate = 5e6
rxGain = 20
txGain = 30
delay = int(1e6)
nsamps=8092
NumSamples = nsamps*5
reportInterval = 1

rxSerial = "RF3E000069"
#rxSerial = "RF3E000069"
txSerial = "RF3E000075"
txChan = 0
rxChan = 0

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

#Init Buffers
sampsRecv = np.zeros(nsamps, dtype=np.complex64)
sampsSend = np.zeros(nsamps, dtype=np.complex64)
sampsSend = np.random.uniform(0,1,nsamps)+1j*np.random.uniform(0,1,nsamps)
period = 600;
sampsSend = np.exp(2j*np.pi*np.arange(nsamps)/period)


# for sdr in [tx_sdr, rx_sdr]: sdr.setHardwareTime(0, "TRIGGER")
# tx_sdr.writeSetting('SYNC_DELAYS', "")
# tx_sdr.writeSetting("TRIGGER_GEN", "")

for sdr in [tx_sdr, rx_sdr]: sdr.setHardwareTime(0)


#txProtoPulse = np.exp((1j*2*np.pi*freq/rate)*np.arange(nsamps)).astype(np.complex64)

#Init Streams
rxStream = rx_sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0], {})
txStream = tx_sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0], {})

ts = tx_sdr.getHardwareTime() + delay
tx_sdr.activateStream(txStream, SOAPY_SDR_HAS_TIME, ts)
timeOfTxStreamActivation = time.time()
rx_sdr.activateStream(rxStream, SOAPY_SDR_HAS_TIME, ts)
timeOfRxStreamActivation = time.time()
	
timeOfLastPrintout=time.time()
print("Commencing TX/RX")
total_samps = 0
rxBuffs = np.array([], np.complex64)
txVec = np.array([], np.complex64)
while total_samps < NumSamples:
	
	if total_samps >= 0:
		timeSinceTxStreamActivation = (time.time() - timeOfTxStreamActivation)*1e6
		print("Start writing stream at %f us from stream activation" % timeSinceTxStreamActivation)
	sr = tx_sdr.writeStream(txStream, [sampsSend], nsamps)
	if sr.ret != nsamps:
		raise Exception('tx fail - Bad write')
	if total_samps >= 0:
		timeSinceTxSteamActivation = (time.time() - timeOfTxStreamActivation)*1e6
		print("Done writing stream at %f us from stream activation" % timeSinceTxStreamActivation)

	#do some tx processing here
	txVec = np.concatenate((txVec, sampsSend[:sr.ret]))
	
	#timeSinceRxSteamActivation = (time.time() - timeOfRxStreamActivation)*1e6
	#timeSinceRxStreamTrigger = delay*1e-3 - timeSinceRxSteamActivation
	#print("Reading stream at %f us from stream activation, %f us from stream trigger" 
	#	% (timeSinceRxSteamActivation, timeSinceRxStreamTrigger))	


	if total_samps >= 0:
		timeSinceRxStreamActivation = (time.time() - timeOfRxStreamActivation)*1e6
		print("Start reading stream at %f us from stream activation" % timeSinceRxStreamActivation)
	sr = rx_sdr.readStream(rxStream, [sampsRecv], nsamps, timeoutUs=int(10e6))		
	if sr.ret != nsamps:
		raise Exception('receive fail - Bad Read')
	if total_samps >= 0:
		timeSinceRxStreamActivation = (time.time() - timeOfRxStreamActivation)*1e6
		print("Done reading stream at %f us from stream activation" % timeSinceRxStreamActivation)
	#do some rx processing here
	rxBuffs = np.concatenate((rxBuffs, sampsRecv[:sr.ret]))

	
	total_samps += nsamps
	if time.time() - timeOfLastPrintout > reportInterval:
		timeOfLastPrintout=time.time()
		print("Total Samples Sent and Received: %i" % total_samps)
	
	#It is probably good to sleep here, but the readStream will block sufficiently
	#it just depends on your processing


waveFormFig, waveFormAxList = plt.subplots(3,1,sharex=True)
waveFormAxList[0].plot(np.real(txVec),'b')
waveFormAxList[0].plot(np.imag(txVec),'r')
#waveFormAxList[0].set_title('Desired Transmit')
waveFormAxList[1].plot(np.real(rxBuffs),'b')
waveFormAxList[1].plot(np.imag(rxBuffs),'r')
#[1].set_title('Received Data')


#cleanup streams
print("Cleanup streams")
tx_sdr.deactivateStream(txStream)
rx_sdr.deactivateStream(rxStream)
rx_sdr.closeStream(rxStream)
tx_sdr.closeStream(txStream)

print("Done!")

plt.show()