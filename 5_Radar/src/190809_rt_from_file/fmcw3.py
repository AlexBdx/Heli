# import pylibftdi as ftdi
from queue import Queue  # , Empty
# from threading import Thread
import datetime
import time
import argparse
# import os
import tqdm
from math import ceil, floor
import os

from fmcw import *


def main():
    # Create ADC and FPGA objects
    adf4158 = adc.ADF4158()
    fmcw3 = fmcw.FMCW3(adf4158, encoding=ENCODING)

    # Set up objects
    fmcw3.set_gpio(led=True, adf_ce=True)
    fmcw3.set_adc(oe2=True)
    fmcw3.clear_adc(oe1=True, shdn1=True, shdn2=True)
    delay = fmcw3.set_sweep(s['f0'], s['bw'], s['t_sweep'], s['t_delay'])  # Returned value delay not used
    fmcw3.set_downsampler(enable=s['down_sampler'], quarter=s['quarter'])
    fmcw3.write_sweep_timer(s['t_sweep'])
    fmcw3.write_sweep_delay(s['t_delay'])
    fmcw3.write_decimate(s['acquisition_decimate'])
    fmcw3.write_pa_off_timer(s['t_delay'] - s['pa_off_advance'])
    fmcw3.clear_gpio(pa_off=True)
    fmcw3.clear_buffer()
    fmcw3.set_channels(a=s['channel_1_active'], b=s['channel_2_active'])  # Would not scale up

    # Create the log and get ready to write
    q = Queue()
    q.put(str(s)+'\n')  # Put the dict as a string to the queue

    timeout = 10*DURATION  # [s] Timeout for reading the queue
    writer = fmcw.Writer(PATH_LOG, q, ENCODING, timeout)  # Automatically overwritten
    writer.start()  # Starts it on a separate thread

    # Calculate how much data is supposed to be written to file
    data_per_frame = s['t_sweep'] * adc_sampling_frequency * ADC_BYTES * s['channel_count']
    frame_period = (s['acquisition_decimate']+1)*(s['t_sweep']+s['t_delay'])
    expected_data = BYTE_USB_READ*ceil(floor(data_per_frame*s['duration']/frame_period)/BYTE_USB_READ) + len(str(s)) + 1
    progress_bar = tqdm.tqdm(total=expected_data)
    progress_bar.update(len(str(s)))  # Add the initial settings
    try:
        t0 = time.perf_counter()  # Keep track of starting time
        while time.perf_counter() - t0 < s['duration']:
            r = fmcw3.device.read(BYTE_USB_READ)
            #print(type(r))
            #print(bytearray(r, encoding=ENCODING))

            if len(r) != 0:
                q.put(r)
                progress_bar.update(len(r))
    finally:
        fmcw3.set_adc(oe1=True, shdn1=True, shdn2=True)
        fmcw3.set_channels(a=False, b=False)
        fmcw3.clear_gpio(led=True, adf_ce=True)
        fmcw3.set_gpio(pa_off=True)
        fmcw3.clear_buffer()
        fmcw3.close()
        progress_bar.close()
        q.put('')
        writer.join()
    print("[INFO] Expected {:,} byte".format(expected_data))


# I. Parameters setup
# I.1. Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--duration", type=int, default=2, help="duration of the recording [s]")
args = vars(ap.parse_args())
DURATION = args["duration"]
PATH_LOG = 'fmcw3.log'
ENCODING = 'latin1'

# I.2. [USER] USER PARAMETERS.
# f0 [Hz] | Starting frequency of the chirp
# bw [Hz] | Bandwidth used for the chirp
# t_sweep [s] | Duration of the chirp/sweep
# t_delay [s] | Delay between two chirps/sweeps
# pa_off_advance [s] | NOT SURE YET
# acquisition_decimate [-] | Sweeps to skip. 0 means no sweeps are skipped
# down_sampler [-] | Divides the ADC clock by 2
# quarter [-] | Divides the ADC clock by 4

s = {
    'f0': 5.3e9,
    'bw': 300e6,
    't_sweep': 1e-3,
    't_delay': 2e-3,
    'pa_off_advance': 0.2e-3,
    'acquisition_decimate': 2,
    'down_sampler': True,
    'quarter': False
}
active_channels = {
    'channel_1_active': True,
    'channel_2_active': True,
    'channel_3_active': False,
    'channel_4_active': False
}  # Add all the channels you want


# I.3. [AUTO] Other parameters
# Finalize the settings
s['duration'] = DURATION
s['timestamp'] = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
s['channel_count'] = sum(active_channels.values())  # Add the channel count
s = {**s, **active_channels}  # Merge both dict
print("[INFO] Acquisition parameters:")
for item in s.items():
    print(item)

# Set some other parameters based on hardware and design
s['if_amplifier_bandwidth'] = 2e6  # [Hz] Nothing we can do about that?
s['fir_gain'] = 9.0
s['adc_ref'] = 1  # [V] Reference voltage on the ADC inputs
s['adc_bits'] = 12
s['d_antenna'] = 28e-3  # [m] Distance between the two antennas. Corresponds to lambda/2 for f0 = 5.3 GHz
s['angle_limit'] = 55
s['angle_pad'] = 100
s['c'] = 299792458.0  # [m/s]
s['channel_offset'] = 21  # Not sure what that is
ADC_BITS = 12
ADC_BYTES = ADC_BITS//8 + 1
adc_sampling_frequency = 1e6  # [Hz] Effective sampling rate for the ADC as the IF amplifier has a 2 MHz bandwidth
#BYTE_USB_READ = 0x10000
BYTE_USB_READ = 0x10000

# II. Run Acquisition loop
if __name__ == '__main__':
    print("[INFO] Process set to niceness", os.nice(0))  # Only root can be below 0
    main()