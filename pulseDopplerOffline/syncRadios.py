import SoapySDR
import time
from SoapySDR import * 

if __name__ == '__main__':
	txSerial = "RF3E000069"
	rxSerial = "RF3E000075"

	tx_sdr = SoapySDR.Device(dict(driver="iris", serial = txSerial))
	rx_sdr = SoapySDR.Device(dict(driver="iris", serial = rxSerial))

	# td = (rx_sdr.getHardwareTime() - tx_sdr.getHardwareTime())*1e-9
	# print("time offset before sync:%e seconds" % td)

	time.sleep(1)

	print("\nTime at each radio:")
	print("Trigger     radio time: %d ns" %  tx_sdr.getHardwareTime())
	print("Non-trigger radio time: %d ns" %  rx_sdr.getHardwareTime())


	tx_sdr.writeSetting('SYNC_DELAYS', "")
	for sdr in [tx_sdr, rx_sdr]: sdr.setHardwareTime(0, "TRIGGER")
	tx_sdr.writeSetting("TRIGGER_GEN", "")
	
	time.sleep(1)

	print("\nTime at each radio ~1s after sync attempted:")
	print("Trigger     radio time: %d ns" %  tx_sdr.getHardwareTime())
	print("Non-trigger radio time: %d ns <-- was not reset" %  rx_sdr.getHardwareTime())

