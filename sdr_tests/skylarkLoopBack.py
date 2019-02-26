########################################################################
## Measure round trip delay through RF loopback/leakage
########################################################################

import SoapySDR
from SoapySDR import * #SOAPY_SDR_ constants
import numpy as np
from scipy import signal
from optparse import OptionParser
import time
import os
import matplotlib.pyplot as plt
import pickle

# Define SDR parameters
serials = ["RF3E000075","RF3E000069"]
txChan = 0
rxChan = 0
txGain = 50 # 60 No Loopback, 20 + 30dB Attenuator for Loopback
rxGain = 40
rate = 10e6
freq = 2500e6
txAnt = "TRX"
rxAnt = "TRX"

# Configure SDRs
tx_sdr = SoapySDR.Device(dict(driver="iris", serial = serials[0]))
if not tx_sdr.hasHardwareTime():
    raise Exception('this device does not support timed streaming')

rx_sdr = SoapySDR.Device(dict(driver="iris", serial = serials[1]))
if not tx_sdr.hasHardwareTime():
    raise Exception('this device does not support timed streaming')

#set sample rate
rx_sdr.setSampleRate(SOAPY_SDR_RX, rxChan, rate)
tx_sdr.setSampleRate(SOAPY_SDR_TX, txChan, rate)
print("Actual Rx Rate %f Msps"%(rx_sdr.getSampleRate(SOAPY_SDR_RX, rxChan)/1e6))
print("Actual Tx Rate %f Msps"%(tx_sdr.getSampleRate(SOAPY_SDR_TX, txChan)/1e6))

#set antenna
rx_sdr.setAntenna(SOAPY_SDR_RX, rxChan, rxAnt)
tx_sdr.setAntenna(SOAPY_SDR_TX, txChan, txAnt)

#set overall gain
rx_sdr.setGain(SOAPY_SDR_RX, rxChan, rxGain)
tx_sdr.setGain(SOAPY_SDR_TX, txChan, txGain)

#tune frontends
rx_sdr.setFrequency(SOAPY_SDR_RX, rxChan, freq)
tx_sdr.setFrequency(SOAPY_SDR_TX, txChan, freq)

# Build pulse train
header_len = 140
header = np.array([0.0]*header_len).astype(np.complex64)
footer_len = 40
footer = np.array([0.0]*footer_len).astype(np.complex64)

# Sinc proto-signal
# pulseDuration = 500
# pulseWidth = 5 
# pulseAmplitude = 0.3
# x = np.linspace(-pulseWidth, pulseWidth, pulseDuration)
# txProtoPulse = pulseAmplitude*signal.hilbert(np.sinc(x)).astype(np.complex64)

pulsePeriod = 100 # Number of samples between pulses

pulseFile = open("skylark_transmit2.txt", 'r')
for line in pulseFile:
    input_interleaved = np.fromstring(line, dtype=np.float32, count=-1, sep=' ')
    sigLength = int(len(input_interleaved) / 2)
    interleavedValues = input_interleaved.reshape((sigLength, 2))
    realSamples = interleavedValues[:, 0]
    imagSamples = interleavedValues[:, 1]
    inputbuffer = realSamples + 1j * imagSamples
txProtoPulse = inputbuffer
# txProtoPulse = np.concatenate((inputbuffer, np.array([0.0]*pulsePeriod).astype(np.complex64)))
# freq = 1e6
# txProtoPulse = np.exp((1j*2*np.pi*freq/rate)*np.arange(80)).astype(np.complex64)

# numPulses = 1*1024
numPulses = 10
txPulse = header
for ix in range(numPulses):
    txPulse = np.concatenate((txPulse,txProtoPulse))

numTxSamps = len(txPulse)
numRxSamps = numTxSamps+100

# Synchronize the SDRs
for sdr in [tx_sdr, rx_sdr]: sdr.setHardwareTime(0, "TRIGGER")
tx_sdr.writeSetting('SYNC_DELAYS', "")
# tx_sdr.writeSetting("TRIGGER_GEN", "")

#normalize the samples
def normalize(samps):
    samps = samps - np.mean(samps) #remove dc
    samps = np.absolute(samps) #magnitude
    samps = samps/max(samps) #norm ampl to peak
    #print samps[:100]
    return samps

def cfloat2uint32(arr, order='IQ'):
        arr_i = (np.real(arr) * 32767).astype(np.uint16)
        arr_q = (np.imag(arr) * 32767).astype(np.uint16)
        if order == 'IQ':
            return np.bitwise_or(arr_q ,np.left_shift(arr_i.astype(np.uint32), 16))
        else:
            return np.bitwise_or(arr_i ,np.left_shift(arr_q.astype(np.uint32), 16))


def txRx(sampleOffset): 


    #create rx and tx streams
    print("Create Rx and Tx streams")
    rxStream = rx_sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [rxChan])
    txStream = tx_sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [txChan])

    #let things settle
    time.sleep(1)

    #transmit a pulse in the near future
    tx_sdr.activateStream(txStream)
    txTime0 = tx_sdr.getHardwareTime() + long(0.1e9) #100ms
    # txFlags = SOAPY_SDR_HAS_TIME | SOAPY_SDR_END_BURST
    txFlags = SOAPY_SDR_HAS_TIME
    sr = tx_sdr.writeStream(txStream, [txPulse], len(txPulse), txFlags, txTime0)
    if sr.ret != len(txPulse): raise Exception('transmit failed %s'%str(sr))

    #receive slightly before transmit time
    rxBuffs = np.array([], np.complex64)
    rxFlags = SOAPY_SDR_HAS_TIME
    #half of the samples come before the transmit time
    receiveTime = txTime0
    rx_sdr.activateStream(rxStream, rxFlags, receiveTime)
    rxTime0 = None

    #accumulate receive buffer into large contiguous buffer
    while True:
        rxBuff = np.array([0]*1024, np.complex64)
        timeoutUs = long(5e5) #500 ms >> stream time
        sr = rx_sdr.readStream(rxStream, [rxBuff], len(rxBuff), timeoutUs=timeoutUs)

        #stash time on first buffer
        if sr.ret > 0 and len(rxBuffs) == 0:
            rxTime0 = sr.timeNs
            if (sr.flags & SOAPY_SDR_HAS_TIME) == 0:
                raise Exception('receive fail - no timestamp on first readStream %s'%(str(sr)))

        print("Receive Buffer Length: {}, Total Samples Received: {}".format(sr.ret, len(rxBuffs)))
        if len(rxBuffs) > 1e6: break

        #accumulate buffer or exit loop
        if sr.ret > 0: 
            rxBuffs = np.concatenate((rxBuffs, rxBuff[:sr.ret]))
        else: 
            raise Exception('receive fail - Bad Read')
            break

    #cleanup streams
    print("Cleanup streams")
    tx_sdr.deactivateStream(txStream)
    rx_sdr.deactivateStream(rxStream)
    rx_sdr.closeStream(rxStream)
    tx_sdr.closeStream(txStream)

    NumSamples = 2048
    waveFormFig, waveFormAxList = plt.subplots(3,1)
    waveFormAxList[0].plot(np.real(txPulse[0:NumSamples]),'b')
    waveFormAxList[0].plot(np.imag(txPulse[0:NumSamples]),'r')
    waveFormAxList[0].set_title('Desired Transmit')
    waveFormAxList[1].plot(np.real(rxBuffs[0:NumSamples]),'b')
    waveFormAxList[1].plot(np.imag(rxBuffs[0:NumSamples]),'r')
    waveFormAxList[1].set_title('Received Data')
    waveFormAxList[2].plot(np.abs(np.convolve(txPulse[0:NumSamples],txProtoPulse[::-1].conjugate(),mode='same')),'r',label='Transmit')
    waveFormAxList[2].plot(np.abs(np.convolve(rxBuffs[0:NumSamples],txProtoPulse[::-1].conjugate(),mode='same')),'b',label='Receive')
    waveFormAxList[2].set_title('Match Filter')
    waveFormAxList[2].legend()

    #check resulting buffer
    # if len(rxBuffs) != numRxSamps:
    #     raise Exception('receive fail - captured samples %d out of %d'%(len(rxBuffs), numRxSamps))
    if rxTime0 is None:
        raise Exception('receive fail - no valid timestamp')

    plt.show()



def continuousTxRx(dwellDuration): 

    # dwellDuration: Length of the collected pulse train in seconds

    #create rx and tx streams
    print("Create Rx and Tx streams")
    rxStream = rx_sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [rxChan])
    txStream = tx_sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [txChan])

    #let things settle
    time.sleep(1)

    #clear out socket buffer from old requests
    rxBuff = np.array([0]*1024, np.complex64)
    sr = rx_sdr.readStream(rxStream, [rxBuff], len(rxBuff), timeoutUs = 0)
    while sr.ret != SOAPY_SDR_TIMEOUT:
        rxBuff = np.array([0]*1024, np.complex64)
        sr = rx_sdr.readStream(rxStream, [rxBuff], len(rxBuff), timeoutUs = 0)

    #write proto pulse to buffer
    tx_sdr.writeRegisters('TX_RAM_A', 0, cfloat2uint32(txProtoPulse).tolist())

    #receive a waveform at the same time
    # flags = SOAPY_SDR_WAIT_TRIGGER | SOAPY_SDR_END_BURST
    flags = SOAPY_SDR_WAIT_TRIGGER
    rx_sdr.activateStream(rxStream, flags)

    #trigger SDR
    tx_sdr.writeSetting("TX_REPLAY", str(len(txProtoPulse)))
    time.sleep(0.1)
    tx_sdr.writeSetting("TRIGGER_GEN", "")
    time.sleep(0.05) 

    #accumulate receive buffer into large contiguous buffer
    rxBuffs = np.array([], np.complex64)
    rxTime0 = None
    while len(rxBuffs) < int(dwellDuration*rate):
        rxBuff = np.array([0]*1024, np.complex64)
        timeoutUs = long(5e5) #500 ms >> stream time
        sr = rx_sdr.readStream(rxStream, [rxBuff], len(rxBuff), timeoutUs=timeoutUs)

        #stash time on first buffer
        if sr.ret > 0 and len(rxBuffs) == 0:
            rxTime0 = sr.timeNs
            if (sr.flags & SOAPY_SDR_HAS_TIME) == 0:
                raise Exception('receive fail - no timestamp on first readStream %s'%(str(sr)))

        #print("Receive Buffer Length: {}, Time Elapsed: {}".format(sr.ret, len(rxBuffs)/rate))

        #accumulate buffer or exit loop
        if sr.ret > 0: rxBuffs = np.concatenate((rxBuffs, np.conj(rxBuff[:sr.ret])))
        else: break

    print("Expected Number of Pulses Received: {}".format(int(np.floor(len(rxBuffs)/len(txProtoPulse)))))

    #cleanup streams
    print("Cleanup streams")
    tx_sdr.writeSetting("TX_REPLAY", '')
    tx_sdr.deactivateStream(txStream)
    rx_sdr.deactivateStream(rxStream)
    rx_sdr.closeStream(rxStream)
    tx_sdr.closeStream(txStream)

    freqRx = np.fft.fftfreq(len(rxBuffs), 1/rate)
    freq_index = np.argsort(freqRx)
    freqRx = freqRx[freq_index]
    fftRx = np.fft.fft(rxBuffs)[freq_index]

    NumSamps = 2048
    waveFormFig, waveFormAxList = plt.subplots(4,1)
    waveFormAxList[0].plot(np.real(txPulse[0:NumSamps]),'b')
    waveFormAxList[0].plot(np.imag(txPulse[0:NumSamps]),'r')
    waveFormAxList[0].set_title('Desired Transmit')
    waveFormAxList[1].plot(np.real(rxBuffs[0:NumSamps]),'b')
    waveFormAxList[1].plot(np.imag(rxBuffs[0:NumSamps]),'r')
    waveFormAxList[1].set_title('Received Data')
    waveFormAxList[2].plot(np.abs(np.convolve(txPulse[0:NumSamps],txProtoPulse[::-1].conjugate(),mode='same')),'r',label='Transmit')
    waveFormAxList[2].plot(np.abs(np.convolve(rxBuffs[0:NumSamps],txProtoPulse[::-1].conjugate(),mode='same')),'b',label='Receive')
    waveFormAxList[2].set_title('Match Filter')
    waveFormAxList[2].legend()
    waveFormAxList[3].plot(freqRx/1e6, 20*np.log10(np.abs(fftRx)),'b')

    #check resulting buffer
    # if len(rxBuffs) != numRxSamps:
    #     raise Exception('receive fail - captured samples %d out of %d'%(len(rxBuffs), numRxSamps))
    if rxTime0 is None:
        raise Exception('receive fail - no valid timestamp')

    # print("Saving Data")
    # data = rxBuffs
    # sigLength = len(data)
    # interleave_data = np.zeros((sigLength, 2), dtype=np.float32)
    # interleave_data[:,0] = np.real(data)
    # interleave_data[:,1] = np.imag(data)
    # interleave_data = interleave_data.flatten()
    # filename = os.getcwd() + "/skylarkDataCollect/CollectedData/" + "recvData_" + time.strftime("%Y%m%d-%H%M%S") + ".csv"
    # textdata = ' '.join([str(samp) for samp in interleave_data]) + '\n'
    # with open(filename,'w') as fn:
    #     fn.write(textdata)

    print("Done!")

    plt.show()