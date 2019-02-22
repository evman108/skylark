

# Synchronize the SDRs
trig_sdr.writeSetting('SYNC_DELAYS', "")
for sdr in sdrs: sdr.setHardwareTime(0, "TRIGGER")
trig_sdr.writeSetting("TRIGGER_GEN", "")


#create rx streams
rxStreams = [sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0], {"remote:prot":"tcp", "remote:mtu":"1024"}) for sdr in rx_sdrs]
num_rx_r = len(rx_sdrs)*2
sampsRecv = [np.empty(num_samps).astype(np.complex64) for r in range(num_rx_r)]
print("Receiving chunks of %i" % len(sampsRecv[0]))

#create tx stream
txStreams = []
for sdr in tx_sdrs:
	txStream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [0], {"REPLAY": 'true'})
	sdr.activateStream(txStream)
	txStreams.append(txStream)


#create our own sinusoid
if inputbuffer is None:
	Ts = 1/rate
	#s_length = 768*4
	s_length = num_samps
	s_freq = 1e6
	s_time_vals = np.array(np.arange(0,s_length)).transpose()*Ts
	s = np.exp(s_time_vals*1j*2*np.pi*s_freq).astype(np.complex64)*.25
	num_tx_r = len(tx_sdrs)*2
	sampsToSend = [np.zeros(int(num_samps/4)).astype(np.complex64) for r in range(num_tx_r)]
else:
	s = inputbuffer
	s_length = len(s)
	num_tx_r = len(tx_sdrs)*2
	print(int(num_samps/4))
	sampsToSend = [np.zeros(int(num_samps/num_pulses)).astype(np.complex64) for r in range(num_tx_r)]

#samples to send is a two channel array of complex floats
for r in range(num_tx_r):
	print('R-',r)
	sampsToSend[r] = s
	print(sampsToSend)

print("Done initializing")


#clear out socket buffer from old requests
for r,sdr in enumerate(rx_sdrs):
	rxStream = rxStreams[r]
	sr = sdr.readStream(rxStream, [sampsRecv[r*2][:], sampsRecv[r*2+1][:]], len(sampsRecv[0]), timeoutUs = 0)
	while sr.ret != SOAPY_SDR_TIMEOUT:
		sr = sdr.readStream(rxStream, [sampsRecv[r*2][:], sampsRecv[r*2+1][:]], len(sampsRecv[0]), timeoutUs = 0)

time.sleep(0.5)
print("Done clearing buffers")

# #################################################################################
# tx_sdrs[0].writeRegisters('TX_RAM_A', 0, cfloat2uint32(inputbuffer).tolist())

# time.sleep(0.1)

# #receive a waveform at the same time
# for r,sdr in enumerate(rx_sdrs):
# 	rxStream = rxStreams[r]
# 	flags = SOAPY_SDR_WAIT_TRIGGER | SOAPY_SDR_END_BURST
# 	#flags = SOAPY_SDR_HAS_TIME | SOAPY_SDR_END_BURST
# 	sdr.activateStream(rxStream, flags, 0, len(sampsRecv[0]))

# trig_sdr.writeSetting("TRIGGER_GEN", "")
# tx_sdrs[0].writeSetting("TX_REPLAY", str(len(inputbuffer)))
# ################################################################


########################
print(trig_sdr.getHardwareTime())
txTime0 = trig_sdr.getHardwareTime() + long(2e9) #200ms
for t,sdr in enumerate(tx_sdrs):
	txStream = txStreams[t]
	sdr.activateStream(txStream)

	flags = SOAPY_SDR_HAS_TIME | SOAPY_SDR_END_BURST

	numSent = 0
	while numSent < len(sampsToSend[0]):
		print(numSent)
		sr = sdr.writeStream(txStream, [sampsToSend[t*2][numSent:], sampsToSend[t*2+1][numSent:]], len(sampsToSend[0])-numSent, flags)#, timeNs=txTime0)
	
	if sr.ret != len(inputbuffer): raise Exception('transmit failed %s'%str(sr))
print("Done with tx writes")

for r,sdr in enumerate(rx_sdrs):
	rxStream = rxStreams[r]
	flags = SOAPY_SDR_HAS_TIME | SOAPY_SDR_END_BURST
	sdr.activateStream(rxStream, flags, txTime0, len(sampsRecv[0]))
print("Done with rx activates")
##############################


#time.sleep(.5)

for r,sdr in enumerate(rx_sdrs):
	
	rxStream = rxStreams[r]
	sr = sdr.readStream(rxStream, [sampsRecv[r*2], sampsRecv[r*2+1]], len(sampsRecv[0]), timeoutUs=int(1e6))
	if sr.ret != len(sampsRecv[0]):
		print("Bad read!!!")
		
	#remove residual DC offset
	sampsRecv[r*2][:] -= np.mean(sampsRecv[r*2][:])
	sampsRecv[r*2+1][:] -= np.mean(sampsRecv[r*2+1][:])

#look at any async messages
print('Issues:')
for r,sdr in enumerate(tx_sdrs):
	txStream = txStreams[r]
	sr = sdr.readStreamStatus(txStream, timeoutUs=int(1e6))
	print(sr)

#cleanup streams
print("Cleanup streams")
for r,sdr in enumerate(tx_sdrs):
	sdr.deactivateStream(txStreams[r])
	sdr.closeStream(txStreams[r])
#for sdr,rxStream in (rx_sdrs,rxStreams):
for r,sdr in enumerate(rx_sdrs):
	sdr.deactivateStream(rxStreams[r])
	sdr.closeStream(rxStreams[r])
print("Done!")



t = np.arange(len(sampsToSend))
waveFormFig, waveformAxList = plt.subplots(3,1)
waveformAxList[0].plot(t,np.real(sampsToSend[0]))
waveformAxList[0].plot(t,np.imag(sampsToSend[0]))
t = np.arange(sampsRecv[0].size)
waveformAxList[1].plot(t,np.real(sampsRecv[0]))
waveformAxList[1].plot(t,np.imag(sampsRecv[0]))
waveformAxList[2].plot(t,np.real(sampsRecv[1]))
waveformAxList[2].plot(t,np.imag(sampsRecv[1]))

plt.show()




# #accumulate receive buffer into large contiguous buffer
# while True:
#     rxBuff = np.array([0]*1024, np.complex64)
#     timeoutUs = long(5e5) #500 ms >> stream time
#     sr = sdr.readStream(rxStream, [rxBuff], len(rxBuff), timeoutUs=timeoutUs)

#     #stash time on first buffer
#     if sr.ret > 0 and len(rxBuffs) == 0:
#         rxTime0 = sr.timeNs
#         if (sr.flags & SOAPY_SDR_HAS_TIME) == 0:
#             raise Exception('receive fail - no timestamp on first readStream %s'%(str(sr)))

#     #accumulate buffer or exit loop
#     if sr.ret > 0: rxBuffs = np.concatenate((rxBuffs, rxBuff[:sr.ret]))
#     else: break