#!/usr/bin/python3

import numpy as np
import sys
import SoapySDR
import time
from SoapySDR import * #SOAPY_SDR_constants
import matplotlib.pyplot as plt

#if __name__ == '__main__':
freq = 2.45e9
rate = 5e6
rxGain = 10
txGain = 40
delay = int(10e6)
nsamps=8092
NumSamples = nsamps*5
txrxSerial = "RF3E000069"

sdr = SoapySDR.Device(dict(driver="iris", serial = txrxSerial))

# Init Rx
sdr.setSampleRate(SOAPY_SDR_RX, 0, rate)
sdr.setFrequency(SOAPY_SDR_RX, 0, "RF", freq)
sdr.setGain(SOAPY_SDR_RX, 0, rxGain)
sdr.setAntenna(SOAPY_SDR_RX, 0, "TRX")
sdr.setFrequency(SOAPY_SDR_RX, 0, "BB", 0) #don't use cordic
sdr.setDCOffsetMode(SOAPY_SDR_RX, 0, False) #dc removal on rx #we'll remove this in post-processing

# Init Tx
sdr.setSampleRate(SOAPY_SDR_TX, 0, rate)
sdr.setFrequency(SOAPY_SDR_TX, 0, "RF", freq)
sdr.setGain(SOAPY_SDR_TX, 0, txGain)
sdr.setAntenna(SOAPY_SDR_TX, 0, "TRX")

#Init Buffers
sampsRecv = np.zeros(nsamps, dtype=np.complex64)
#sampsSend = np.zeros(nsamps, dtype=np.complex64)
Ts = 1/rate
s_freq = 1e6
s_time_vals = np.array(np.arange(0,nsamps)).transpose()*Ts
sampsSend = np.exp(s_time_vals*1j*2*np.pi*s_freq).astype(np.complex64)*.25

# period = 600;
# sampsSend = np.exp(2j*np.pi*np.arange(nsamps)/period)*.25

#Init Streams
rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0], {})
txStream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0], {})

ts = sdr.getHardwareTime() + delay
sdr.activateStream(txStream, SOAPY_SDR_HAS_TIME, ts)
sdr.activateStream(rxStream, SOAPY_SDR_HAS_TIME, ts)
	
t=time.time()
total_samps = 0
rxBuffs = np.array([], np.complex64)
txVec = np.array([], np.complex64)
while total_samps < NumSamples:
	sr = sdr.writeStream(txStream, [sampsSend], nsamps)
	if sr.ret != nsamps:
		print("Bad Write!!!")
	txVec = np.concatenate((txVec, sampsSend[:sr.ret]))
		
	sr = sdr.readStream(rxStream, [sampsRecv], nsamps, timeoutUs=int(10e6))		
	if sr.ret != nsamps:
		print("Bad Read!!!")
	rxBuffs = np.concatenate((rxBuffs, sampsRecv[:sr.ret]))
	
	total_samps += nsamps
	if time.time() - t > 1:
		t=time.time()
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
sdr.deactivateStream(txStream)
sdr.deactivateStream(rxStream)
sdr.closeStream(rxStream)
sdr.closeStream(txStream)

print("Done!")

plt.show()