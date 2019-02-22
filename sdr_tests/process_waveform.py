import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
import copy

from anglecoord_cpp import AngleCoord
from shorad.datagen.radarreturn import TargetReturnTruth
from shorad.state import RUVState
from shorad.coord import RadarSpaceType
from shorad.quantizer import ZeroIncludeQuantizer
from shorad.rawchanneldata import RawChannelData
from shorad.waveform import LfmWaveform
from shorad.visualize.visualize_rd_map import plotRdMap
from shorad import test_objects as tob


class SDRInterface():

    NUM_QUANTIZATION_BITS = 12
    SIM_TARGET_RANGE = 300.0

    def __init__(self, waveform: LfmWaveform, useQuantization=False):

        self._waveform = waveform

        self._quantizer = None
        if useQuantization:
            self._quantizer = ZeroIncludeQuantizer(
                SDRInterface.NUM_QUANTIZATION_BITS, 2.0)

    def getTransmitString(self) -> str:
        samples = self._getWaveformSamples(True)
        return self._waveformSampToString(samples)

    def simulateExampleReturn(self) -> str:

        # Simulate target
        executive = tob.getShoradExecutive()
        antenna = tob.getShoradAntenna()
        radarSpace = executive.getRadarSpace(
            time=0.0, spaceType=RadarSpaceType.RUV, waveform=self._waveform)
        targetMean = np.array([SDRInterface.SIM_TARGET_RANGE, 0.0, 0.0, 0.0])
        target = tob.getTestTarget(targetMean=targetMean, antenna=antenna,
                                   waveform=self._waveform)
        entityState = target.getEntityState(0.0)
        ruvState = RUVState.toRUVState(entityState.position, radarSpace)
        rcs = (0.1, 0.1)
        returnTruth = TargetReturnTruth(ruvState, entityState, rcs, target)
        returnTruthDict = {1: returnTruth}

        # Syntheize received waveform
        receiver = tob.getShoradReceiver()
        receiver.setRandomState(np.random.RandomState())
        receiverResult = receiver.genAllSubarraySignals(
            antenna, self._waveform, returnTruthDict, AngleCoord(0.0, 0.0))

        # Return string from a single channel
        return self._waveformSampToString(receiverResult.uuPlusVvPlusChannel)

    def processReceivedSamples(self, sampString: str, toBaseband: bool):

        receivedSamp = sdrInterface._parseReceivedSamples(receivedString)
        if toBaseband:
            self._shiftToBaseband(receivedSamp)

        self._generateRDMap(receivedSamp)
        self._plotTimeAndFreq(receivedSamp)

    def plotTransmitPulse(self) -> None:
        samples = self._getWaveformSamples(False)
        self._plotTimeAndFreq(samples)

    def _generateRDMap(self, receivedSamp: np.ndarray) -> None:

        # Create a RawChannelData with the input data used in each channel
        rawChannels = RawChannelData(
            uuPlusVvPlusChannel=receivedSamp,
            uuMinusVvPlusChannel=copy.deepcopy(receivedSamp),
            uuPlusVvMinusChannel=copy.deepcopy(receivedSamp),
            uuMinusVvMinusChannel=copy.deepcopy(receivedSamp))

        noisePower = tob.getShoradAntenna().processedNoisePower(self._waveform)
        sigproc = tob.getShoradSignalProcessor()
        processedChannels = sigproc.processChannels(rawChannels, self._waveform,
                                                    noisePower)
        plotRdMap(processedChannels.rdMap(), self._waveform, (8, 8), True)

    def _plotTimeAndFreq(self, receivedSamp: np.ndarray) -> None:

        # Plot time series data
        plt.figure()
        timeSamples = np.arange(len(receivedSamp)) * self._waveform.sampPeriod
        plt.plot(timeSamples, receivedSamp.real)
        plt.plot(timeSamples, receivedSamp.imag)
        plt.title('Waveform Samples')
        plt.xlabel('Time (s)')

        # Plot frequency spectrum
        spectrum = np.abs(np.fft.fft(receivedSamp)**2)
        freqArray = np.fft.fftfreq(receivedSamp.size, self._waveform.sampPeriod)
        idx = np.argsort(freqArray)

        plt.figure()
        plt.plot(freqArray[idx] / 1e6, spectrum[idx])
        plt.title('Frequency Spectrum')
        plt.xlabel('Frequency (MHz)')

    def _waveformSampToString(self, samples: np.ndarray) -> str:
        realSampleArray = samples.real
        imagSampleArray = samples.imag

        if self._quantizer is not None:
            realSampleArray = self._getQuantizedSamples(realSampleArray)
            imagSampleArray = self._getQuantizedSamples(imagSampleArray)

        sampMatrix = np.array([realSampleArray, imagSampleArray])
        interleavedSamples = sampMatrix.T.flatten()

        return ' '.join([str(samp) for samp in interleavedSamples]) + '\n'

    def _getQuantizedSamples(self, samples: np.ndarray) -> np.ndarray:
        samples = self._quantizer.quantize(samples)
        samples /= self._quantizer.resolution
        return samples.astype(np.int64)

    def _getWaveformSamples(self, useTrain: bool) -> np.ndarray:
        samples = self._waveform.transmitSignal(useTrain=useTrain)
        self._shiftToBaseband(samples)
        return samples

    def _shiftToBaseband(self, samples: np.ndarray) -> None:
        samples *= np.exp(-2j * np.pi * self._waveform.intermediateFreq *
                          self._waveform.sampPeriod *
                          np.arange(0, len(samples)))

    def _parseReceivedSamples(self, sampString: str) -> np.ndarray:
        interleavedValues = np.fromstring(sampString, dtype=np.float64,
                                          count=-1, sep=' ')
        # return interleavedValues.astype(np.float32).view(np.complex128)
        sigLength = int(len(interleavedValues) / 2)
        interleavedValues = interleavedValues.reshape((sigLength, 2))
        realSamples = interleavedValues[:, 0]
        imagSamples = interleavedValues[:, 1]

        if self._quantizer is not None:
            realSamples *= self._quantizer.resolution
            imagSamples *= self._quantizer.resolution

        return (realSamples + 1j * imagSamples).astype(np.complex128)


if __name__ == '__main__':

    '''
    Output transmit waveform to stdout:
    python util/skylark_backend_test.py --transmit <args>

    Process received waveform via stdin
    <instream> | python util/skylark_backend_test.py <args>

    Generate simulated received waveform for testing backend processing
    python util/skylark_backend_test.py --simulate <args>
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--quantize', action='store_true')
    parser.add_argument('--transmit', action='store_true')
    parser.add_argument('--simulate', action='store_true')
    parser.add_argument('--waveform', type=str)
    args = parser.parse_args()

    if args.waveform == 'neotech':
        toBaseband = True
        waveform = tob.getShoradSearchWaveform()

    else:  # Skylark
        toBaseband = False
        waveform = tob.getShoradSearchWaveform(
            rfFreq=2.5e9, bandwidth=5e6, numPulses=256, pulseWidth=50e-6,
            pri=120e-6, windowType='Hamming', sampFreq=10e6,
            intermediateFreq=0e6)

    sdrInterface = SDRInterface(waveform=waveform,
                                useQuantization=args.quantize)

    if args.transmit:
        transmitString = sdrInterface.getTransmitString()
        sys.stdout.write(transmitString)
        sdrInterface.plotTransmitPulse()
        plt.show()

    elif args.simulate:
        simulatedReturn = sdrInterface.simulateExampleReturn()
        sys.stdout.write(simulatedReturn)

    else:
        receivedString = sys.stdin.readline()
        sys.stdin = open('/dev/tty')  # so pdb can work after stdin read
        sdrInterface.processReceivedSamples(receivedString, toBaseband)
        plt.show()
