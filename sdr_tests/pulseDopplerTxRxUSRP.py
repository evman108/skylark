#!/usr/bin/python
#
#	Demonstrate MIMO capability on 2+ Iris-030s connected to each other (sharing clocks and a trigger).
#	Half of boards are designated as TX and send complex sines in a TDMA style.
#	Other half of boards are RX and plot received signal (the sinusoids from each TX radio).
#
#	--LTSMode uses one board, not in the array, to send an LTS followed by a sine wave (one from each antenna).
#	Boards in an array receive on both antennas and align the signal to the LTS and display a plot per antenna.
#
#	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#	INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
#	PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
#	FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#	OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#	DEALINGS IN THE SOFTWARE.
#
#	(c) info@skylarkwireless.com 2016

from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

import SoapySDR
from SoapySDR import * #SOAPY_SDR_ constants
import numpy as np
from optparse import OptionParser
import time
import os
import math
import lts
import sys

num_pulses = 16

def cfloat2uint32(arr, order='IQ'):
		arr_i = (np.real(arr) * 32767).astype(np.uint16)
		arr_q = (np.imag(arr) * 32767).astype(np.uint16)
		if order == 'IQ':
			return np.bitwise_or(arr_q ,np.left_shift(arr_i.astype(np.uint32), 16))
		else:
			return np.bitwise_or(arr_i ,np.left_shift(arr_q.astype(np.uint32), 16))
	
def uint32tocfloat(arr, order='IQ'):
	arr_hi = ((np.right_shift(arr, 16).astype(np.int16))/32768.0)
	arr_lo = (np.bitwise_and(arr, 0xFFFF).astype(np.int16))/32768.0
	if order == 'IQ':
		return (arr_hi + 1j*arr_lo).astype(np.complex64)
	else:
		return (arr_lo + 1j*arr_hi).astype(np.complex64)

class MIMO_SDR:
	'''
		Class that initializes 2+ Irises (based on the serials provided),
		makes half tx and half rx, then sets up the tx radios to send TDMA-style
		complex sinusoids.  Provides getSamples() to perform a tx/rx.
		
		Assumes the Irises are connected to each other and sharing a trigger.
		The first serial provided is the trigger Iris (highest on the chain).
	'''
	
	def __init__(self,
		args,
		rate,
		freq=None,
		bw=None,
		txGain=None,
		rxGain=None,
		rxAnt=None,
		txAnt=None,
		serials=None,
		num_samps=None,
		LTSMode=False,
		ShowConst=False,
		inputbuffer=None,
	):

		self.sdrs = [SoapySDR.Device(dict(driver="uhd", serial = serial)) for serial in serials]
		for serial in serials:
			print(serial)
		#	SoapySDR.Device(dict(driver="iris", serial = serial))

		self.tx_sdrs = self.sdrs[0:len(self.sdrs)//2]
		self.rx_sdrs = self.sdrs[len(self.sdrs)//2:]
		self.trig_sdr = self.sdrs[0]
		self.num_samps = num_samps
		self.LTSMode = LTSMode
		self.ShowConst = ShowConst
		self.inputbuffer = inputbuffer
		self.rate = rate

		print("Using %i tx Irises and %i rx Irises." % (len(self.tx_sdrs), len(self.rx_sdrs)) )

		#override default settings
		for sdr in self.sdrs:
			for chan in [0]:
				print(chan)
				print(rxAnt)
				print(txAnt)
				print(freq)
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
		
				# NO DOCUMENTATION ON THESE SETTINGS
				#sdr.writeSetting(SOAPY_SDR_RX, chan, 'CALIBRATE', 'SKLK')
				#sdr.writeSetting(SOAPY_SDR_TX, chan, 'CALIBRATE', 'SKLK')
				#sdr.writeSetting('SPI_TDD_MODE', 'MIMO')

		# NO DOCUMENTATION ON THESE SETTINGS
		self.trig_sdr.writeSetting('SYNC_DELAYS', "")
		for sdr in self.sdrs: sdr.setHardwareTime(0, "TRIGGER")


		#create rx streams
		self.rxStreams = [sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0], {"remote:prot":"tcp", "remote:mtu":"1024"}) for sdr in self.rx_sdrs]
		num_rx_r = len(self.rx_sdrs)*2
		self.sampsRecv = [np.empty(num_samps).astype(np.complex64) for r in range(num_rx_r)]
		print("Receiving chunks of %i" % len(self.sampsRecv[0]))

		#create tx stream
		self.txStreams = []
		for sdr in self.tx_sdrs:
			txStream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0], {"REPLAY": 'true'})
			sdr.activateStream(txStream)
			self.txStreams.append(txStream)
	

		#create our own sinusoid
		if inputbuffer is None:
			Ts = 1/rate
			s_length = 768*4
			s_freq = 1e6
			s_time_vals = np.array(np.arange(0,s_length)).transpose()*Ts
			s = np.exp(s_time_vals*1j*2*np.pi*s_freq).astype(np.complex64)*.25
			num_tx_r = len(self.tx_sdrs)*2
			self.sampsToSend = [np.zeros(int(num_samps/4)).astype(np.complex64) for r in range(num_tx_r)]
		else:
			s = inputbuffer
			s_length = len(s)
			num_tx_r = len(self.tx_sdrs)*2
			print(int(num_samps/4))
			self.sampsToSend = [np.zeros(int(num_samps/num_pulses)).astype(np.complex64) for r in range(num_tx_r)]

		#samples to send is a two channel array of complex floats
		for r in range(num_tx_r):
			self.sampsToSend[r] = s

		print("Done initializing MIMOGui.")


	def getSamples(self):
		#print("getSamples()")
		#return self.sampsToSend

		#clear out socket buffer from old requests
		for r,sdr in enumerate(self.rx_sdrs):
			rxStream = self.rxStreams[r]
			sr = sdr.readStream(rxStream, [self.sampsRecv[r*2][:], self.sampsRecv[r*2+1][:]], len(self.sampsRecv[0]), timeoutUs = 0)
			while sr.ret != SOAPY_SDR_TIMEOUT:
				sr = sdr.readStream(rxStream, [self.sampsRecv[r*2][:], self.sampsRecv[r*2+1][:]], len(self.sampsRecv[0]), timeoutUs = 0)

		flags = SOAPY_SDR_WAIT_TRIGGER | SOAPY_SDR_END_BURST

		#self.tx_sdrs[0].writeRegisters('TX_RAM_A', 0, cfloat2uint32(self.inputbuffer).tolist())

		#transmit and receive at this time in the future
		for r,sdr in enumerate(self.tx_sdrs):
			print("Setting up tx streams")
			txStream = self.txStreams[r]
			sdr.activateStream(txStream)
			numSent = 0
			while numSent < num_samps:
				sr = sdr.writeStream(txStream, [self.sampsToSend[r*2][numSent:], self.sampsToSend[r*2+1][numSent:]], self.num_samps-numSent, flags)
				print(sr) 
				#assertGreater(sr.ret, 0)
				numSent += sr.ret
				if sr.ret == -1:
					print('Bad Write!')
					return -1

		time.sleep(0.1)

		#receive a waveform at the same time
		for r,sdr in enumerate(self.rx_sdrs):
			rxStream = self.rxStreams[r]
			flags = SOAPY_SDR_WAIT_TRIGGER | SOAPY_SDR_END_BURST
			#flags = SOAPY_SDR_HAS_TIME | SOAPY_SDR_END_BURST
			sdr.activateStream(rxStream, flags, 0, len(self.sampsRecv[0]))

		self.trig_sdr.writeSetting("TRIGGER_GEN", "")
		#self.tx_sdrs[0].writeSetting("TX_REPLAY", str(len(self.inputbuffer)))

		#time.sleep(.5)

		for r,sdr in enumerate(self.rx_sdrs):
			
			rxStream = self.rxStreams[r]
			sr = sdr.readStream(rxStream, [self.sampsRecv[r*2], self.sampsRecv[r*2+1]], len(self.sampsRecv[0]), timeoutUs=int(1e6))
			if sr.ret != len(self.sampsRecv[0]):
				print("Bad read!!!")
				
			#remove residual DC offset
			self.sampsRecv[r*2][:] -= np.mean(self.sampsRecv[r*2][:])
			self.sampsRecv[r*2+1][:] -= np.mean(self.sampsRecv[r*2+1][:])
		
		#look at any async messages
		print('Issues:')
		for r,sdr in enumerate(self.tx_sdrs):
			txStream = self.txStreams[r]
			sr = sdr.readStreamStatus(txStream, timeoutUs=int(1e6))
			print(sr)


		ret_buffer = self.sampsRecv[0]
		ret_buffer_interleaved = np.array([ret_buffer.real, ret_buffer.imag]).T.flatten()
		ret_buffer_str = ' '.join([str(samp) for samp in ret_buffer_interleaved]) + '\n'
		sys.stderr.write(ret_buffer_str)
		ret_buffer = self.sampsRecv[1]
		ret_buffer_interleaved = np.array([ret_buffer.real, ret_buffer.imag]).T.flatten()
		ret_buffer_str = ' '.join([str(samp) for samp in ret_buffer_interleaved]) + '\n'
		sys.stderr.write(ret_buffer_str)

		return self.sampsRecv

	def mimo_test_close(self):
		#cleanup streams
		print("Cleanup streams")
		for r,sdr in enumerate(self.tx_sdrs):
			sdr.deactivateStream(self.txStreams[r])
			sdr.closeStream(self.txStreams[r])
		#for sdr,rxStream in (rx_sdrs,rxStreams):
		for r,sdr in enumerate(self.rx_sdrs):
			sdr.deactivateStream(self.rxStreams[r])
			sdr.closeStream(self.rxStreams[r])
		print("Done!")
	
class QT_GUI:
	'''
		This is probably not appropriate QT style, but it works fine for now.
		
		This class just keeps track of the graphic elements in pyqtgraph.
		It setups up a grid of plots (1 column if 2 radios, otherwise 2 columns),
		then simply plots the received IQ samples of each RX radio on every update in
		the corresponding plot.
	'''

	class myWin(pg.GraphicsWindow):
		'''Handle key presses.'''
		def keyPressEvent(self, event):
			#print(event.key())
			if event.key() == QtCore.Qt.Key_Escape:
				self.showNormal()
			elif event.key() == ord('f') or event.key() == ord('F'):
				self.showFullScreen() 
			elif event.key() == ord('q') or event.key() == ord('Q'):
				self.timer.stop() #todo: timer and cleanup should be passed in the constructor, not set after creation.
				self.cleanup() #weird, this crashes things -- it's like Soapy doesn't detect it's already closed and tries to free it again.
				sys.exit(self.app.exec_())
				#self.closeEvent(event)
			return super(QT_GUI.myWin, self).keyPressEvent(event)
		def closeEvent(self, evnt):
			self.timer.stop() #todo: timer and cleanup should be passed in the constructor, not set after creation.
			#self.cleanup() #weird, this crashes things -- it's like Soapy doesn't detect it's already closed and tries to free it again.
			super(QT_GUI.myWin, self).closeEvent(evnt)
			
	def update(self):
		'''Get new samples (from mimo_sdr) and plot them.'''

		samps = self.mimo_sdr.getSamples()
		if samps == -1: 
			return #don't die -- just try again on the next update

		for plt in range(self.num_plots):
			#if we're in LTSMode we align the samples by detecting the LTS
			self.I_plots[plt].setData(samps[plt][:self.num_samps].real) # - np.mean(samps[plt][:self.num_samps].real))
			self.Q_plots[plt].setData(samps[plt][:self.num_samps].imag) # - np.mean(samps[plt][:self.num_samps].imag))
				#self.I_plots[plt].setData(self.mimo_sdr.sampsToSend[plt][:self.num_samps].real)
				#self.Q_plots[plt].setData(self.mimo_sdr.sampsToSend[plt][:self.num_samps].imag)
				
	def __init__(self, num_plots=None, num_samps=4096, update_interval=1000, mimo_sdr=None, LTSMode=False, ShowConst=False):
		self.num_plots = num_plots
		self.num_samps = num_samps
		self.mimo_sdr = mimo_sdr
		self.LTSMode = LTSMode
		self.ShowConst = ShowConst
		print("Number of Samps")
		print(num_samps)
		

		#QtGui.QApplication.setGraphicsSystem('raster')
		app = QtGui.QApplication([])
		#mw = QtGui.QMainWindow()
		#mw.resize(800,800)

		'''  Window arrangement settings '''

		#num_plots = 2 #len(sdr_serials) #this is /2 since have are tx, then *2 since each has 2 ant
		num_cols = 2 if num_plots > 2 or self.ShowConst else 1
		num_rows = int(math.ceil(num_plots/2))
		plt_scale = .6

		win = self.myWin(title="Skylark Wireless | Iris MIMO Demo")
		win.cleanup = self.mimo_sdr.mimo_test_close
		#win.resize(1000,600)
		#win.showMaximized()
		win.showFullScreen() #To return from full-screen mode, call showNormal().
		# Enable antialiasing for prettier plots
		pg.setConfigOptions(antialias=True)
		win.ci.layout.setRowMaximumHeight(0,80)


		vb = win.addViewBox(col=0, colspan=num_cols, lockAspect=True, enableMouse=False, invertY=True) #, border='00ff00'
		vb.setBackgroundColor('ffffff')
		#vb.setStyle() #todo:make rounded corners This takes a QStyle
		img = QtGui.QGraphicsPixmapItem(QtGui.QPixmap('data/logo.tif'))
		vb.addItem(img)
		#vb.scaleBy(4)

		IQ_plots = [None]*num_plots
		I_plots = [None]*num_plots
		Q_plots = [None]*num_plots
		Corr_plots = [None]*num_plots
		Const_plots = [None]*num_plots
		Const_data = [None]*num_plots
		LR = [None]*num_plots
		for plt in range(num_plots):
			if plt % num_cols == 0 or self.ShowConst:
				win.nextRow()

			IQ_plots[plt] = win.addPlot(title="IQ Samples Rx %i" % (plt+1))
			IQ_plots[plt].setTitle('<span style="font-size: 22pt;">IQ Samples Rx %i</span>' % (plt+1))
			IQ_plots[plt].setRange(xRange=(0,num_samps),yRange=[-plt_scale,plt_scale],disableAutoRange=True)
			I_plots[plt] = IQ_plots[plt].plot(pen=(255,0,0), name="I")
			Q_plots[plt] = IQ_plots[plt].plot(pen=(0,0,255), name="Q")
			Corr_plots[plt] = IQ_plots[plt].plot(pen=(155,155,0), name="Corr")

			#shade packet.
			LR[plt] = pg.LinearRegionItem()
			LR[plt].setZValue(100)
			#IQ_plots[plt].addItem(LR[plt])

			if self.ShowConst:
				Const_plots[plt] = win.addPlot(title='Constellation %i' % (plt+1))
				Const_plots[plt].setTitle('<span style="font-size: 22pt;">Constellation %i</span>' % (plt+1))
				Const_plots[plt].setRange(xRange=[-.5,.5],yRange=[-.5,.5],disableAutoRange=True)
				Const_plots[plt].setAspectLocked(lock=True, ratio=1)
				Const_data[plt] = pg.ScatterPlotItem(size=2, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
				Const_plots[plt].addItem(Const_data[plt])

		self.IQ_plots = IQ_plots
		self.I_plots = I_plots
		self.Q_plots = Q_plots
		self.Corr_plots = Corr_plots
		self.Const_plots = Const_plots
		self.Const_data = Const_data
		self.LR = LR

		timer = QtCore.QTimer()
		timer.timeout.connect(self.update)
		timer.start(update_interval)
		win.timer = timer

		#this is critical!  (otherwise it breaks, since if we don't keep a reference they are deleted: 
		#http://stackoverflow.com/questions/5339062/python-pyside-internal-c-object-already-deleted)
		self.timer = timer
		self.app = app
		self.win = win
		self.win.app = app

		


if __name__ == '__main__':

	parser = OptionParser()
	parser.add_option("--args", type="string", dest="args", help="device factor arguments", default="")
	parser.add_option("--rate", type="float", dest="rate", help="Sample rate", default=5e6)
	parser.add_option("--rxAnt", type="string", dest="rxAnt", help="Optional Rx antenna (RX or TRX)", default="RX2")
	parser.add_option("--txAnt", type="string", dest="txAnt", help="Optional Tx antenna (TRX)", default="TX/RX")
	parser.add_option("--txGain", type="float", dest="txGain", help="Optional Tx gain (dB)", default=100.0)
	parser.add_option("--rxGain", type="float", dest="rxGain", help="Optional Rx gain (dB)", default=80.0)
	parser.add_option("--freq", type="float", dest="freq", help="Optional Tx freq (Hz)", default=5800e6) #2484e6) #563e6
	parser.add_option("--bw", type="float", dest="bw", help="Optional filter bw (Hz)", default=None)
	parser.add_option("--serials", type=str, dest="serials", help="SDR Serial Numbers, e.g. 00002 00004", default="31119E8 31119E8") #"0127 0125 0120 0113 0111 0106 0107")
	parser.add_option("--LTSMode", action="store_true", dest="LTSMode", help="LTSMode (Use last radio as standalone in TxReplay mode, then receive on all radios on the array.)", default=False)
	parser.add_option("--Constellation", action="store_true", dest="ShowConst", help="Send OFDM packets and decode/display constellation.", default=False)
	
	(options, args) = parser.parse_args()
	print(args)
	
	serials = options.serials.split()
	num_sdrs=len(serials)
	num_samps=1024*num_sdrs*4
	num_plots = (num_sdrs-1)*2 if options.LTSMode else num_sdrs
	print(num_sdrs)
	
	pulseFile = open("skylark_transmit.txt", 'r')
	for line in pulseFile:
		input_interleaved = np.fromstring(line, dtype=np.float32, count=-1, sep=' ')
		sigLength = int(len(input_interleaved) / 2)
		interleavedValues = input_interleaved.reshape((sigLength, 2))
		realSamples = interleavedValues[:, 0]
		imagSamples = interleavedValues[:, 1]
		inputbuffer = realSamples + 1j * imagSamples

	num_samps = num_pulses*len(inputbuffer)
	print(num_samps)
	
		#ret_buffer = test_run.transmitAndReceive(input_buffer)
	
		# interleave complex buffer
		#ret_buffer_interleaved = np.array([ret_buffer.real, ret_buffer.imag]).T.flatten()
		#ret_buffer_str = ' '.join([str(samp) for samp in ret_buffer_interleaved]) + '\n'
		#sys.stdout.write(ret_buffer_str)


	mimo_sdr = MIMO_SDR(
		args=options.args,
		rate=options.rate,
		freq=options.freq,
		bw=options.bw,
		rxAnt=options.rxAnt,
		txAnt=options.txAnt,
		txGain=options.txGain,
		rxGain=options.rxGain,
		serials=serials,
		num_samps=num_samps,
		LTSMode=options.LTSMode,
		ShowConst=options.ShowConst,
		inputbuffer=inputbuffer,
	)
	qt_gui = QT_GUI(num_plots=num_plots, num_samps=num_samps, mimo_sdr=mimo_sdr, LTSMode=options.LTSMode, ShowConst=options.ShowConst)
	
	qt_gui.update()
	#qt_gui.app.exec_()
	## Start Qt event loop unless running in interactive mode or using pyside.
	import sys
	if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
		QtGui.QApplication.instance().exec_()

	#python MIMOGui.py --serials="00002 00004"
