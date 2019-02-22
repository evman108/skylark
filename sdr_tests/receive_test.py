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



serials = "31119E8"
rxChan = 0
rxGain = 40 #20
rate = 256e3
# freq = 5800e6
freq = 150e6
rxAnt = "RX2"
#header_len = 140 # @ 10e6 sampling
#header_len = 230 # @ 15e6 sampling <- scales with sampling rate?
# header_len = 50 # @ 18e6 sampling <- scales with sampling rate?
# footer_len = 50 
# header = np.array([0.0]*header_len).astype(np.complex64)
# footer = np.array([0.0]*footer_len).astype(np.complex64)

# pulseFile = open("skylark_transmit.txt", 'r')
# for line in pulseFile:
#     input_interleaved = np.fromstring(line, dtype=np.float32, count=-1, sep=' ')
#     sigLength = int(len(input_interleaved) / 2)
#     interleavedValues = input_interleaved.reshape((sigLength, 2))
#     realSamples = interleavedValues[:, 0]
#     imagSamples = interleavedValues[:, 1]
#     inputbuffer = realSamples + 1j * imagSamples
# txPulse = inputbuffer

# numTxSamps=len(txPulse)+header_len+footer_len
numRxSamps=int(1e6)
# pulseWidth=20
# pulseScaleFactor=0.3
# x = np.linspace(-pulseWidth, pulseWidth, numTxSamps-header_len-footer_len)

# txPulse = signal.hilbert(np.sinc(x)).astype(np.complex64)
# txPulse = np.sinc(x).astype(np.complex64)
# txPulse = np.zeros(len(x)).astype(np.complex64)
# txPulse = np.ones(len(x)).astype(np.complex64)


# txPulse = np.concatenate((header, txPulse*pulseScaleFactor, footer))

# tx_sdr = SoapySDR.Device(dict(driver="uhd", serial = serials[0]))
# if not tx_sdr.hasHardwareTime():
#     raise Exception('this device does not support timed streaming')

rx_sdr = SoapySDR.Device(dict(driver="uhd", serial = serials))
# if not tx_sdr.hasHardwareTime():
#     raise Exception('this device does not support timed streaming')

#set clock rate first
# if clockRate is not None: sdr.setMasterClockRate(clockRate)

#set sample rate
rx_sdr.setSampleRate(SOAPY_SDR_RX, rxChan, rate)
# tx_sdr.setSampleRate(SOAPY_SDR_TX, txChan, rate)
print("Actual Rx Rate %f Msps"%(rx_sdr.getSampleRate(SOAPY_SDR_RX, rxChan)/1e6))
# print("Actual Tx Rate %f Msps"%(tx_sdr.getSampleRate(SOAPY_SDR_TX, txChan)/1e6))

#set antenna
rx_sdr.setAntenna(SOAPY_SDR_RX, rxChan, rxAnt)
# tx_sdr.setAntenna(SOAPY_SDR_TX, txChan, txAnt)

#set overall gain
rx_sdr.setGain(SOAPY_SDR_RX, rxChan, rxGain)
# tx_sdr.setGain(SOAPY_SDR_TX, txChan, txGain)

#tune frontends
rx_sdr.setFrequency(SOAPY_SDR_RX, rxChan, freq)
# tx_sdr.setFrequency(SOAPY_SDR_TX, txChan, freq)



#set bandwidth
#if rxBw is not None: sdr.setBandwidth(SOAPY_SDR_RX, rxChan, rxBw)
#if txBw is not None: sdr.setBandwidth(SOAPY_SDR_TX, txChan, txBw)

# Synchronize the SDRs
# for sdr in [tx_sdr, rx_sdr]: sdr.setHardwareTime(0, "TRIGGER")
# tx_sdr.writeSetting('SYNC_DELAYS', "")
# tx_sdr.writeSetting("TRIGGER_GEN", "")



#normalize the samples
def normalize(samps):
    samps = samps - np.mean(samps) #remove dc
    samps = np.absolute(samps) #magnitude
    samps = samps/max(samps) #norm ampl to peak
    #print samps[:100]
    return samps




def txRx(sampleOffset): 


    #create rx and tx streams
    print("Create Rx and Tx streams")
    rxStream = rx_sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [rxChan])
    # txStream = tx_sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [txChan])

    #let things settle
    time.sleep(1)

    #transmit a pulse in the near future
    # tx_sdr.activateStream(txStream)
    txTime0 = rx_sdr.getHardwareTime() + long(0.1e9) #100ms
    # txFlags = SOAPY_SDR_HAS_TIME | SOAPY_SDR_END_BURST
    # sr = tx_sdr.writeStream(txStream, [txPulse], len(txPulse), txFlags, txTime0)
    # if sr.ret != len(txPulse): raise Exception('transmit failed %s'%str(sr))

    #receive slightly before transmit time
    rxBuffs = np.array([], np.complex64)
    rxFlags = SOAPY_SDR_HAS_TIME | SOAPY_SDR_END_BURST
    #half of the samples come before the transmit time
    receiveTime = txTime0 - long((numRxSamps/rate)*1e9)/2
    rx_sdr.activateStream(rxStream, rxFlags, txTime0, numRxSamps)
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

        #accumulate buffer or exit loop
        if sr.ret > 0: rxBuffs = np.concatenate((rxBuffs, rxBuff[:sr.ret]))
        else: break

    #cleanup streams
    # print("Cleanup streams")
    # tx_sdr.deactivateStream(txStream)
    # rx_sdr.closeStream(rxStream)
    # tx_sdr.closeStream(txStream)

    #check resulting buffer
    if len(rxBuffs) != numRxSamps:
        raise Exception('receive fail - captured samples %d out of %d'%(len(rxBuffs), numRxSamps))
    if rxTime0 is None:
        raise Exception('receive fail - no valid timestamp')

    #clear initial samples because transients
    #rxMean = np.mean(rxBuffs)
    #for i in range(len(rxBuffs)/100): rxBuffs[i] = rxMean

    # txSig = np.append(txPulse, np.zeros(numRxSamps -numTxSamps).astype(np.complex64))
    # print(len(txSig))
    t = np.arange(numRxSamps)
    print(len(t))
    plt.plot(t,np.real(rxBuffs),'b')
    plt.plot(t,np.imag(rxBuffs),'r')
    # waveFormFig, waveformAxList = plt.subplots(1,1)
    # waveformAxList[0].plot(t,np.real(txSig))
    # waveformAxList[0].plot(t,np.imag(txSig))
    # waveformAxList[0].plot(t,np.real(rxBuffs))
    # waveformAxList[0].plot(t,np.imag(rxBuffs))
    # waveformAxList[1].plot(t,normalize(np.convolve(rxBuffs,txPulse[::-1].conjugate(),mode='same')))
    # waveformAxList[1].plot(t,normalize(np.convolve(txSig,txPulse[::-1].conjugate(),mode='same')))
    #print(len(np.convolve(rxBuffs,txPulse,mode='same')))

    # pcNorm = normalize(np.convolve(rxBuffs,txPulse[::-1].conjugate(),mode='same'))
    #txPulseNorm = normalize(txPulse)
    # txPulseNorm = normalize(np.convolve(txSig,txPulse[::-1].conjugate(),mode='same'))
    rxBuffsNorm = normalize(rxBuffs)

    # fftTx = np.fft.fft(txSig)
    # freqTx = np.fft.fftfreq(len(txSig), 1/rate)
    fftRx = np.fft.fft(rxBuffs)
    freqRx = np.fft.fftfreq(len(rxBuffs), 1/rate)
    # fftFig, fftAx = plt.subplots(2,1)
    # fftAx[0].plot(freqTx/1e6, np.abs(fftTx), 'b')
    # fftAx[0].set_title('TX')
    # fftAx[0].set_xticklabels([])
    # fftAx[0].grid()
    # fftAx[1].plot(freqRx/1e6, np.abs(fftRx), 'r')
    # fftAx[1].set_title('RX')
    # fftAx[1].set_xlabel('Frequency (MHz)')
    # fftAx[1].grid()

    # #dump debug samples
    # if dumpDir is not None:
    #     txPulseNorm.tofile(os.path.join(dumpDir, 'txNorm.dat'))
    #     rxBuffsNorm.tofile(os.path.join(dumpDir, 'rxNorm.dat'))
    #     np.real(rxBuffs).tofile(os.path.join(dumpDir, 'rxRawI.dat'))
    #     np.imag(rxBuffs).tofile(os.path.join(dumpDir, 'rxRawQ.dat'))

    # #look for the for peak index for time offsets
    #rxArgmaxIndex = np.argmax(rxBuffsNorm)
    # rxArgmaxIndex = np.argmax(pcNorm)
    # txArgmaxIndex = np.argmax(txPulseNorm)

    #check goodness of peak by comparing argmax and correlation
    #rxCoorIndex = np.argmax(np.correlate(rxBuffsNorm, txPulseNorm))+len(txPulseNorm)/2
    #if abs(rxCoorIndex-rxArgmaxIndex) > len(txPulseNorm)/4:
    #    print('correlation(%d) does not match argmax(%d), probably bad data'%(rxCoorIndex, rxArgmaxIndex))

    #calculate time offset
    # calibration = -4350
    # print(txTime0,rxTime0)
    # txPeakTime = txTime0 + long((txArgmaxIndex/rate)*1e9)
    # rxPeakTime = rxTime0 + long(((rxArgmaxIndex - sampleOffset)/rate)*1e9)
    # timeDelta = rxPeakTime - txPeakTime + calibration
    # rangeDelta = 299792458 * timeDelta / 1e9
    # #print('>>> Sample delta 1: %i'%(rxCoorIndex - txArgmaxIndex))
    # print('>>> RxStart-TxStart: %i'%(rxTime0-txTime0))
    # print('>>> Calibration: %i'%(rxTime0-txTime0 + (rxArgmaxIndex - txArgmaxIndex)/rate*1e9))
    # print('>>> Sample delta: %i'%(rxArgmaxIndex - sampleOffset - txArgmaxIndex))
    # print('>>> Time delta: %.2f us'%(timeDelta/1e3))
    # print('>>> Range delta: %.2f ft'%(rangeDelta*3.28))

    # print("Done!")

    plt.show()
    import pdb; pdb.set_trace()