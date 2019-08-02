import pylibftdi as ftdi
#from adf4158 import ADF4158
#from Queue import Queue, Empty
from queue import Queue, Empty
from threading import Thread
import datetime
import time
import argparse
import os

# Custom packages
from fmcw import *

#print(ftdi.__file__)


# Beginning of the script ------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--duration", type=int, default=1, help="duration of the recording [s]")
args = vars(ap.parse_args())
DURATION = args["duration"]
#DURATION=1

#ENCODING = 'utf-8'
#ENCODING = 'ascii'
ENCODING = 'latin1'

adf4158 = adc.ADF4158()
fmcw3 = fmcw.FMCW3(adf4158, encoding=ENCODING)

f0 = 5.3e9
bw = 600e6
tsweep = 1e-3
tsweep = 1e-3
tdelay = 2e-3
#tdelay = 0
pa_off_advance = 0.2e-3
decimate = 3
#decimate = 1
ch_a = True
ch_b = True
downsampler = True
#downsampler = False
quarter = False

fmcw3.set_gpio(led=True, adf_ce=True)
fmcw3.set_adc(oe2=True)
fmcw3.clear_adc(oe1=True, shdn1=True, shdn2= True)
delay = fmcw3.set_sweep(f0, bw, tsweep, tdelay)

fmcw3.set_downsampler(enable=downsampler, quarter=quarter)
fmcw3.write_sweep_timer(tsweep)
fmcw3.write_sweep_delay(tdelay)
fmcw3.write_decimate(decimate)
fmcw3.write_pa_off_timer(tdelay - pa_off_advance)
fmcw3.clear_gpio(pa_off=True)
fmcw3.clear_buffer()

q = Queue()

date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
ts = datetime.datetime.now().strftime('%y%m%d_%H%M%S')

settings_dict = {'duration': DURATION, 'date': date, 'f0':f0, 'bw': bw, 'tsweep': tsweep, 'tdelay': tdelay, 'a': ch_a, 'b': ch_b, 'decimate': decimate, 'downsampler': downsampler, 'quarter': quarter}
settings_dict = "{}\n".format(settings_dict)

print("[INFO] Settings:\n{}\n".format(settings_dict))
q.put(settings_dict)

"""
# Rebuild the folder that will contain the data
folderName = 'FMCW3_import'
if os.path.isdir(folderName):
    sp.run(['rm', '-r', folderName])
sp.run(['mkdir', folderName])
writer = Writer(ts+'_fmcw3.log', q)
"""
writer = fmcw.Writer('fmcw3.log', q, ENCODING)  # Automatically overwritten
writer.start()

fmcw3.set_channels(a=ch_a, b=ch_b)

# t0 = time.perf_counter()  # Python3
t0 = time.perf_counter()
#print("[INFO] Started at {} s".format(t0))
try:
    while time.perf_counter() - t0 < DURATION:
        #print("[INFO] Elasped time: {:.3f} s".format(time.perf_counter() - t0))
        r = fmcw3.device.read(0x10000)
        print(type(r))
        #print("[INFO] Read {:,} bytes from device".format(len(r)))
        #r = fmcw3.device.readline(5000)
        if len(r) != 0:
            q.put(r)
finally:
    fmcw3.set_adc(oe1=True, shdn1=True, shdn2= True)
    fmcw3.set_channels(a=False, b=False)
    fmcw3.clear_gpio(led=True, adf_ce=True)
    fmcw3.set_gpio(pa_off=True)
    fmcw3.clear_buffer()
    fmcw3.close()
    #print('Done')
    q.put('')
    writer.join()
