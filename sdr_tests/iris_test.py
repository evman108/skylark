#!/usr/bin/python

import numpy as np

import SoapySDR
from SoapySDR import * #SOAPY_SDR_ constants
import numpy as np
from optparse import OptionParser
import time
import os
import sys
import math


class IRIS_Setup:
    '''
        This class initializes 2 Irises (based on the serials provided),
        makes one tx and one rx, then sets up the tx radios to send a
        signal provided on stdin.  Provides getSamples() to perform a tx/rx.
        
        Assumes the Irises are connected to each other and tx SDR does the
        triggering.
    '''
    
    def __init__(self,
        args,
        rate,
        freq=None,
        bw=None,
        txSerial=None,
        rxSerial=None,
        txGain=None,
        rxGain=None,
        rxAnt=None,
        txAnt=None,
    ):

        self.txSDR = SoapySDR.Device(dict(driver="iris", serial = txSerial))
        self.rxSDR = SoapySDR.Device(dict(driver="iris", serial = rxSerial))

        # print("Using Iris: {} for tx and Iris: {} for rx.", txSerial, rxSerial)

        for sdr in [self.txSDR, self.rxSDR]:
            chan = 0
            if rate is not None: sdr.setSampleRate(SOAPY_SDR_RX, chan, rate)
            if bw is not None: sdr.setBandwidth(SOAPY_SDR_RX, chan, bw)
            if rxGain is not None: sdr.setGain(SOAPY_SDR_RX, chan, rxGain)
            if freq is not None: sdr.setFrequency(SOAPY_SDR_RX, chan, "RF", freq)
            if rxAnt is not None: sdr.setAntenna(SOAPY_SDR_RX, chan, rxAnt)
            sdr.setFrequency(SOAPY_SDR_RX, chan, "BB", 0) #don't use cordic
            sdr.setDCOffsetMode(SOAPY_SDR_RX, chan, True) #dc removal on rx

            if rate is not None: sdr.setSampleRate(SOAPY_SDR_TX, chan, rate)
            if bw is not None: sdr.setBandwidth(SOAPY_SDR_TX, chan, bw)
            if txGain is not None: sdr.setGain(SOAPY_SDR_TX, chan, txGain)
            if freq is not None: sdr.setFrequency(SOAPY_SDR_TX, chan, "RF", freq)
            if txAnt is not None: sdr.setAntenna(SOAPY_SDR_TX, chan, txAnt)
            sdr.setFrequency(SOAPY_SDR_TX, chan, "BB", 0) #don't use cordic
            sdr.writeSetting(SOAPY_SDR_RX, chan, 'CALIBRATE', 'SKLK')
            sdr.writeSetting(SOAPY_SDR_TX, chan, 'CALIBRATE', 'SKLK')
            sdr.writeSetting('SPI_TDD_MODE', 'MIMO')

        self.txSDR.writeSetting('SYNC_DELAYS', "")
        self.txSDR.setHardwareTime(0, "TRIGGER")
        self.rxSDR.setHardwareTime(0, "TRIGGER")

        # print("Done with Iris Setup")


    def transmitAndReceive(self, in_buffer):

        self.sampsToSend = in_buffer
	print(self.sampsToSend)
        num_samps = len(self.sampsToSend)

        #create rx streams
        self.rxStream = self.rxSDR.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0], {"remote:prot":"tcp", "remote:mtu":"1024"})
        self.sampsRecv = np.empty(num_samps).astype(np.complex64)
        print("Receiving chunks of %i" % len(self.sampsRecv))

        #create tx stream
        self.txStream = self.txSDR.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0], {})
        self.txSDR.activateStream(self.txStream)
        # print("Activate Tx Stream")


        #clear out socket buffer from old requests
        sr = self.rxSDR.readStream(self.rxStream, self.sampsRecv, len(self.sampsRecv), timeoutUs = 0)
        while sr.ret != SOAPY_SDR_TIMEOUT:
            sr = self.rxSDR.readStream(self.rxStream, self.sampsRecv, len(self.sampsRecv), timeoutUs = 0)

        flags = SOAPY_SDR_WAIT_TRIGGER | SOAPY_SDR_END_BURST

        #transmit and receive at this time in the future
        numSent = 0
        while numSent < num_samps:
            sr = self.txSDR.writeStream(self.txStream, self.sampsToSend[numSent:], num_samps-numSent, flags)
            numSent += sr.ret
	    print('num sent: ', numSent)
            if sr.ret == -1:
                # print('Bad Write!')
                return -1

        #receive a waveform at the same time
        self.rxSDR.activateStream(self.rxStream, flags, 0, len(self.sampsRecv))

        #trigger in the near future
        time.sleep(0.1)
        self.txSDR.writeSetting("TRIGGER_GEN", "")
        time.sleep(0.15)
            
        sr = self.rxSDR.readStream(self.rxStream, self.sampsRecv, len(self.sampsRecv), timeoutUs=int(1e6))
        # if sr.ret != len(self.sampsRecv):
            # print("Bad read!!!")
            
        #remove residual DC offset
	print(self.sampsRecv)
        self.sampsRecv[:] -= np.mean(self.sampsRecv[:])
        # self.sampsRecv[:] -= np.mean(self.sampsRecv[:])
        
        #look at any async messages
        sr = self.txSDR.readStreamStatus(self.txStream, timeoutUs=int(1e6))

        # print("Cleanup streams")
        self.txSDR.deactivateStream(self.txStream)
        self.txSDR.closeStream(self.txStream)

        self.rxSDR.deactivateStream(self.rxStream)
        self.rxSDR.closeStream(self.rxStream)
        # print("Done!")

        return self.sampsRecv


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--args", type="string", dest="args", help="device factor arguments", default="")
    parser.add_option("--rate", type="float", dest="rate", help="Sample rate", default=10e6)
    parser.add_option("--txSerial", type=str, dest="txserial", help="Tx SDR Serial Numbers, e.g. RF3E00000", default="RF3E000069")
    parser.add_option("--rxSerial", type=str, dest="rxserial", help="Rx SDR Serial Numbers, e.g. RF3E00001", default="RF3E000075")
    parser.add_option("--txGain", type="float", dest="txGain", help="Optional Tx gain (dB)", default=40.0)
    parser.add_option("--rxGain", type="float", dest="rxGain", help="Optional Rx gain (dB)", default=30.0)
    parser.add_option("--txAnt", type="string", dest="txAnt", help="Optional Tx antenna (TRX)", default="TRX")
    parser.add_option("--rxAnt", type="string", dest="rxAnt", help="Optional Rx antenna (RX or TRX)", default="TRX")
    parser.add_option("--freq", type="float", dest="freq", help="Optional Tx freq (Hz)", default=3.1e9)
    parser.add_option("--bw", type="float", dest="bw", help="Optional filter bw (Hz)", default=None)
    
    (options, args) = parser.parse_args()
    print(options)

    test_run = IRIS_Setup(
        args = options.args,
        rate = options.rate,
        freq = options.freq,
        bw = options.bw,
        txSerial = options.txserial,
        rxSerial = options.rxserial,
        txGain = options.txGain,
        rxGain = options.rxGain,
        rxAnt = options.rxAnt,
        txAnt = options.txAnt,
    )

    # Read transmit buffer from stdin
    for line in sys.stdin:
        input_interleaved = np.fromstring(line, dtype=np.float32, count=-1, sep=' ')
        input_buffer = input_interleaved.astype(np.float32).view(np.complex64)

        ret_buffer = test_run.transmitAndReceive(input_buffer)

        # interleave complex buffer
        ret_buffer_interleaved = np.array([ret_buffer.real, ret_buffer.imag]).T.flatten()
        ret_buffer_str = ' '.join([str(samp) for samp in ret_buffer_interleaved]) + '\n'
        sys.stdout.write(ret_buffer_str)

