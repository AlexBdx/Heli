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
import numpy as np
import multiprocessing as mp

from fmcw import *


def main():
    if 1:
        # I.3. [AUTO] Hardware and physical constants

        """[TBM]
        # I.4. [AUTO] Rebuild the folder architecture
        if PATH_LOG[0] != '/':  # Make it an absolute path
            PATH_LOG = os.path.join(os.getcwd(), PATH_LOG)
        PATH_FOLDER_LOG = os.path.split(PATH_LOG)[0]
        FOLDER_NAME = os.path.split(PATH_FOLDER_LOG)[1]
        PATH_OUTPUTS = os.path.join(PATH_FOLDER_LOG, OUTPUTS)
    
        if os.path.isdir(PATH_OUTPUTS):
            os.system('rm -r ' + PATH_OUTPUTS)
        os.system('mkdir ' + PATH_OUTPUTS)
        """

        # Create ADC and FPGA objects
        adf4158 = adc.ADF4158()
        fmcw3 = fmcw.FMCW3(adf4158, encoding=s['ENCODING'])

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
        fmcw3.set_channels(a=s[1], b=s[2])  # Would not scale up

        # Create all the bases needed

        """SHOULD BE ABLE TO CALCULATE THESE STRAIGHT OUT OF THE SETTINGS"""
        # Now it is time to plot that crap!
        t, f, d, angles = postprocessing.create_bases(s)
        angle_mask = ~(np.isnan(angles) + (np.abs(angles) > s['angle_limit']))
        #angles_masked = angles[angle_mask]
        print("Max ADC range: {:.1f} m".format(d[-1]))
        tfd_angles = (t, f, d, angles, angle_mask)
        """========================================================================================================"""

        # Create the log and get ready to write
        raw_usb_data_to_file = Queue()
        #raw_usb_data_to_file.put(str(s)+'\n')

        raw_usb_to_decode = Queue()
        #raw_usb_to_decode.put(str(s) + '\n')

        #write_raw_to_file = fmcw.Writer(PATH_LOG, raw_usb_data_to_file, s['ENCODING'], s['timeout'])  # Log is automatically overwritten
        write_raw_to_file = postprocessing.Writer(raw_usb_data_to_file, s, encoding='latin1')  # Log is automatically overwritten
        write_raw_to_file.start()  # Starts it on a separate thread

        decode_raw = postprocessing.Decode(raw_usb_to_decode, s, tfd_angles)
        decode_raw.start()  # Starts it on a separate thread

        # Calculate how much data is supposed to be written to file
        # data_per_frame = s['t_sweep'] * adc_sampling_frequency * ADC_BYTES * s['channel_count']
        # frame_period = (s['acquisition_decimate']+1)*(s['t_sweep']+s['t_delay'])


        """[Data Acquisition]"""
        try:
            t0 = time.perf_counter()  # Keep track of starting time
            while time.perf_counter() - t0 < s['duration']:  # Endless if np.inf
                t1 = time.perf_counter()
                r = fmcw3.device.read(s['BYTE_USB_READ'])
                t2 = time.perf_counter()
                print("Read {} byte in {:.3f} s ({:,} byte/s)".format(len(r), t2-t1, round(len(r)/(t2-t1))))

                if len(r) != 0:
                    raw_usb_data_to_file.put(r)
                    raw_usb_to_decode.put(r)

        finally:
            fmcw3.set_adc(oe1=True, shdn1=True, shdn2=True)
            fmcw3.set_channels(a=False, b=False)
            fmcw3.clear_gpio(led=True, adf_ce=True)
            fmcw3.set_gpio(pa_off=True)
            fmcw3.clear_buffer()
            fmcw3.close()
            raw_usb_data_to_file.put('')
            raw_usb_to_decode.put('')
            write_raw_to_file.join()
            decode_raw.join()

    if 0:

        """CAN BE REMOVED IF THERE IS AN ACQUISITION (REDUNDANT)"""
        # Now it is time to plot that crap!
        t, f, d, angles = postprocessing.create_bases(s)
        angle_mask = ~(np.isnan(angles) + (np.abs(angles) > s['angle_limit']))
        #angles_masked = angles[angle_mask]
        print("Max ADC range: {:.1f} m".format(d[-1]))
        tfd_angles = (t, f, d, angles, angle_mask)
        """========================================================================================================"""


        """[def decode_batch] ======================================================================================"""
        #def process_batch(s):
        # Now we process the file
        read_mode = 'rb' if BINARY else 'r'  # Written as str but read as byte
        
        """
        ch = dict()
        for k in range(nb_channels):
            ch[k + 1] = []
        ch['skipped_sweeps'] = []  # Stores the skipped frames for both channels
        """
        counter_sweeps = 0
        next_header = None
        rest = bytes("", encoding=s['ENCODING'])  # Start without a rest
        with open(os.path.join(PATH_LOG_FOLDER, 'fmcw3.log'), read_mode) as f:
            _ = f.readline()  # Ignore the settings
            print("[INFO] Settings end at", f.tell())
            f.seek(f.tell()+np.random.randint(100))  # Start randomly in a sweep
            print("[INFO] Randomly moved the start of the batch to", f.tell())
            batch = f.read(s['BYTE_USB_READ'])  # Read a batch
        ch, next_header, rest, new_counter_sweeps = postprocessing.process_batch(rest, batch, s, next_header, counter_sweeps, verbose=False)

        # Process the batches we have received
        # II.3. Sanity checks - DO NOT REMOVE
        if np.array_equal(ch[1], ch[2]):  # Test for the down_sampler bug
            raise ValueError('[ERROR] Channel data is identical. Have you checked the downsampler?')
        for channel in ch:  # Test channel shapes
            if type(channel) == int:
                assert ch[1].shape == ch[channel].shape
        assert new_counter_sweeps == len(ch[1]) + counter_sweeps  # Test counter_sweep incrementation
        print("[INFO] Found {} channels of shape {}".format(len(ch) - 1, ch[1].shape))

        # Display results
        expected_lines = ch[1].shape[0]
        skipped_sweeps = len(ch['skipped_sweeps'])
        ratio = 100 * (1 - skipped_sweeps / expected_lines)
        total_duration = s['T'] * ch[1].shape[0]  # Total recording time
        print("[INFO] Found {} frames | Skipped : {} | Success rate: {:.1f} %".format(ch[1].shape[0], skipped_sweeps, ratio))
        print('[INFO] Imported {:.3f} s of data as {} frames of {:.3f} s'.format(total_duration, ch[1].shape[0], s['T']))
        #print('[INFO] Import done in {:.3f} s ({:.1f} s/s)'.format(t1 - t0, total_duration / (t1 - t0)))
        """END OF DECODE BATCH===================================================="""




        """POST PROCESSING SECTION AND DISPLAY - WILL BE MOVED TO SEPARATE PROCESSES"""
        # Create a window
        w = np.kaiser(len(ch[1][0]), s['kaiser_beta'])
        w *= len(w) / np.sum(w)

        # III. Post-processing
        timing = dict()  # [DEBUG]
        if 1:
            if s['subtract_background']:  # [bool]
                ch[1] = postprocessing.subtract_background(ch[1], w, ch)
                ch[2] = postprocessing.subtract_background(ch[2], w, ch)
            elif s['subtract_clutter']:  # [int]
                ch[1] = postprocessing.ssubtract_clutter(ch[1], w, ch, s['subtract_clutter'])
                ch[2] = postprocessing.ssubtract_clutter(ch[2], w, ch, s['subtract_clutter'])
            else:
                ch[1] = w * ch[1]
                ch[2] = w * ch[2]


            # Set up recurring parameters
            clim = None

            # Create the windows
            #if_window = display.if_time_domain_animation(tfd_angles, s, grid=True)
            angle_window = display.angle_animation(tfd_angles, s, method='cross-range')
            max_range_index = int(
                (4 * s['bw'] * (s['SWEEP_LENGTH'] / 2) * s['max_range']) / (s['c'] * s['if_amplifier_bandwidth'] * s['t_sweep']))
            max_range_index = min(max_range_index, s['SWEEP_LENGTH'] // 2)
            #range_time_window = display.range_time_animation(s, max_range_index)
            for index, plot_i in enumerate(tqdm.tqdm(range(len(ch[1])))):
                time_stamp = (None, s['T'] * index, int(s['T'] * index), int(1000 * (s['T'] * index - int(s['T'] * index))))

                # IF DOMAIN OPTIONS
                #if_data, clim = postprocessing.calculate_if_data({1: ch[1][plot_i], 2: ch[2][plot_i]}, s)
                #display.plot_if_time_domain(t, ch, plot_i, s, ylim_if_time_domain, time_stamp, True)
                #if_window.update_plot(if_data, time_stamp[1], clim)

                # ANGLE OPTIONS
                fxdb, clim = postprocessing.calculate_angle_plot({ch[1][plot_i], ch[2][plot_i]}, s, d, clim, angle_mask)
                #display.plot_angle(t, d, fxdb, angles[angle_mask], clim, s['max_range'], time_stamp, method='', show_plot=True)
                angle_window.update_plot(fxdb, time_stamp[1], clim)

                # RANGE TIME OPTIONS
                #im, nb_sweeps, max_range_index, clim = postprocessing.calculate_range_time(ch, s, single_sweep=plot_i)
                #range_time_window.update_plot(im, time_stamp[1], clim)

                time.sleep(0.1)


        """[Final plots]
        # III.4. IF spectrum plot
        if 1:  # [IF spectrum plot]
            
            time_stamp = (None, s['T'] * index, int(s['T'] * index), int(1000 * (s['T'] * index - int(s['T'] * index))))
            display.plot_if_spectrum(d, ch, plot_i, w, s['fir_gain'], s['adc_bits'], time_stamp)
        """
        # III.4. Range time plot
        #total_im = np.array(total_im).transpose()
        im, nb_sweeps, max_range_index, m = postprocessing.calculate_range_time(ch, s)
        #postprocessing.compare_ndarrays(total_im, im)

        #assert np.array_equal(total_im, im)
        if 1:
            f = s['if_amplifier_bandwidth'] / 2.0
            t = np.linspace(0, s['overall_decimate'] * nb_sweeps * (s['t_sweep'] + s['t_delay']), im.shape[1])
            xx, yy = np.meshgrid(
                t,
                np.linspace(0,
                            s['c'] * max_range_index * s['if_amplifier_bandwidth'] / (2 * ch[1].shape[1]) / (
                                        s['bw'] / s['t_sweep']), im.shape[0]))


            time_stamp = (None, s['T'] * index, int(s['T'] * index), int(1000 * (s['T'] * index - int(s['T'] * index))))
            print(t.shape, im.shape)
            display.plot_range_time(t, (xx,yy,im), m, time_stamp, show_plot=True)
            #display.plot_range_time(t, (xx, yy, total_im), m, time_stamp, show_plot=True)
"""========================================================================================================"""


"""START OF THE SCRIPT"""
# I. Parameters setup
# I.1. Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--duration", type=int, default=10, help="duration of the recording [s]")
ap.add_argument("-l", "--log", type=str, default=os.getcwd(), help="Path to folder containing the output logs")
args = vars(ap.parse_args())
DURATION = args["duration"]
if DURATION == 0:
    DURATION = np.inf  # Will run endlessly until keyboard interrupt
PATH_LOG_FOLDER = args["log"]
ENCODING = 'latin1'
BINARY = True


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
    'bw': 600e6,
    't_sweep': 1e-3,
    't_delay': 2e-3,
    'pa_off_advance': 0.2e-3,
    'acquisition_decimate': 2,
    'down_sampler': True,
    'quarter': False,
    'BYTE_USB_READ': 0x10000,
    'range_time_to_display': 1,
    'max_range': 50,
    'channel_offset': 21,
    'swap_chs': True,
    'sweeps_to_read': None,
    'sweeps_to_drop': 0,
    'kaiser_beta': 6,
    'subtract_background': False,
    'subtract_clutter': 0,
    'flag_Hanning': True,
    'real_time_recall': 10,
    'ENCODING': ENCODING,
    'timeout': 0.5,
    'path_raw_log': os.path.join(PATH_LOG_FOLDER, 'fmcw3.log'),
    'path_csv_log': os.path.join(PATH_LOG_FOLDER, 'fmcw3.csv'),
    'refresh_period': 0.5
}
active_channels = {
    1: True,
    2: True,
    3: False,
    4: False
}  # Add all the channels you want

# I.3. [AUTO] Other parameters
# Finalize the settings
s['duration'] = DURATION
s['timestamp'] = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
s['total_channels'] = len(active_channels)
s['channel_count'] = sum(active_channels.values())  # Add the channel count
s['active_channels'] = [key for key in active_channels if active_channels[key]]
print(s['active_channels'])
assert len(s['active_channels'])==2
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
ADC_BITS = 12
ADC_BYTES = ADC_BITS//8 + 1
adc_sampling_frequency = 1e6  # [Hz] Effective sampling rate for the ADC as the IF amplifier has a 2 MHz bandwidth

if s['down_sampler']:
    s['if_amplifier_bandwidth'] /= 2
    if s['quarter']:
        s['if_amplifier_bandwidth'] /= 2
SWEEP_LENGTH = int(s['t_sweep'] * s['if_amplifier_bandwidth'])  # In number of data points
NBYTES_SWEEP = s['channel_count'] * int(s['t_sweep'] * s['if_amplifier_bandwidth']) * ADC_BYTES
s['SWEEP_LENGTH'] = SWEEP_LENGTH
s['NBYTES_SWEEP'] = NBYTES_SWEEP
s['MAX_DIFFERENTIAL_VOLTAGE'] = 1
s['start'] = b'\x7f'[0]
s['overall_decimate'] = (s['sweeps_to_drop'] + 1) * (s['acquisition_decimate'] + 1)
s['T'] = (s['t_sweep'] + s['t_delay']) * s['overall_decimate']  # Period of the meaningful data
if s['refresh_period'] >= s['T']:
    s['refresh_stride'] = round(s['refresh_period']/s['T'])
else:
    s['refresh_stride'] = 1  # Most likely unsustainable if T low
# Sanity check
print()
print("[INFO] Batch size {} byte".format(s['BYTE_USB_READ']))
if s['BYTE_USB_READ'] < 2*s['NBYTES_SWEEP']:  # Otherwise it might be impossible to find a whole frame
    raise ValueError('[ERROR] Need at least to import 2 sweeps per batch')

# II. Run Acquisition loop
if __name__ == '__main__':
    print("[INFO] Process set to niceness", os.nice(0))  # Only root can be below 0
    main()
