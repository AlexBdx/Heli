import numpy as np
import tqdm
import argparse
import os
import time
from fmcw import *


# I. PARAMETERS
# I.1. [USER] Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--log", type=str, default='fmcw3.log', help="Path to log file")
ap.add_argument("-d", "--directory", type=str, default='Outputs', help="Path to output directory")
args = vars(ap.parse_args())
PATH_LOG = args["log"]
OUTPUTS = args["directory"]

# I.2. [USER] Set parameters
max_range = 250  # [m] Requested max range. Actual range depends on ADC capabilities
# channel_dl = 0e-3  # Channel length difference
swap_chs = True
sweeps_to_read = None
sweeps_to_drop = 2
kaiser_beta = 6  # Hanning window
saving_period = 0.5  # [s] How often the different images are saved

subtract_background = False
subtract_clutter = 0  # if > 0, subtract the average of the previous subtract_clutter plots
flag_Hanning = True  # Apply a Hanning window to the peak

# I.3. [AUTO] Hardware and physical constants

MIN_IF_VOLTAGE = -1.1  # [V] Differential input min
MAX_IF_VOLTAGE = 1.1  # [V] Differential input max

# I.4. [AUTO] Rebuild the folder architecture
if PATH_LOG[0] != '/':  # Make it an absolute path
    PATH_LOG = os.path.join(os.getcwd(), PATH_LOG)
PATH_FOLDER_LOG = os.path.split(PATH_LOG)[0]
FOLDER_NAME = os.path.split(PATH_FOLDER_LOG)[1]
PATH_OUTPUTS = os.path.join(PATH_FOLDER_LOG, OUTPUTS)

if os.path.isdir(PATH_OUTPUTS):
    os.system('rm -r '+PATH_OUTPUTS)
os.system('mkdir '+PATH_OUTPUTS)


# II. Import data from log file
BINARY = True
ENCODING = 'latin1' if BINARY else None
start = b'\x7f' if BINARY else '\x7f'
read_mode = 'rb' if BINARY else 'r'
with open(PATH_LOG, read_mode) as f:
    # II.1. Read settings
    s = postprocessing.read_settings(f, encoding=ENCODING)  # Read the setting dictionary
    print("[INFO] {} byte of setting data found:".format(f.tell()))
    for k in s.items():
        print(k)

    # II.1. Set a bunch of parameters
    if s['down_sampler']:
        s['if_amplifier_bandwidth'] /= 2
        if s['quarter']:
            s['if_amplifier_bandwidth'] /= 2
    NBYTES_SWEEP = s['channel_count'] * int(s['t_sweep'] * s['if_amplifier_bandwidth']) * (s['adc_bits']//8 + 1)

    # fc = s['f0'] + s['bw']/2  # [Hz] Center frequency


    print("[INFO] Sweep length: {} byte | Samples in sweep: {}, {} per channel"
          .format(NBYTES_SWEEP, NBYTES_SWEEP//s['channel_count'], NBYTES_SWEEP//s['channel_count']))

    if sweeps_to_read is not None:
        samples = (2 + NBYTES_SWEEP)*sweeps_to_read
    else:
        samples = None

    # II.1. Find the first valid start signal
    current_position, first_frame_number = postprocessing.find_start(f, start, NBYTES_SWEEP)
    print('\n[INFO] Found a start signal at', f.tell())

    # II.1. Import the channel data from the log
    t0 = time.perf_counter()
    ch = postprocessing.import_data(f, start, first_frame_number, NBYTES_SWEEP, samples, sweeps_to_drop, s['channel_count'])
    t1 = time.perf_counter()

# II.3. Sanity check
# CRITICAL, DO NOT REMOVE THIS ASSERTION
if np.array_equal(ch[1], ch[2]):  # Should not need that!!
    raise ValueError('[ERROR] Channel data is identical. Have you checked the downsampler?')
for channel in ch:
    if type(channel) == int:
        assert ch[1].shape == ch[channel].shape
print("[INFO] Found {} channels of shape {}".format(len(ch)-1, ch[1].shape))

# Display results
overall_decimate = (sweeps_to_drop+1)*(s['acquisition_decimate']+1)
T = (s['t_sweep']+s['t_delay'])*overall_decimate  # Period of the meaningful data
expected_lines = ch[1].shape[0]
skipped_sweeps = len(ch['skipped_sweeps'])
ratio = 100*(1 - skipped_sweeps/expected_lines)
total_duration = T*ch[1].shape[0]  # Total recording time

index_to_save = []
nb_plot_to_save = int(s['duration']/saving_period) + 1  # 0 is always saved, hence the +1
for k in range(nb_plot_to_save):
    index_to_save.append(int(k*saving_period / T))
    # print(index_to_save[-1]*T)
# print(index_to_save)
print("[INFO] Found {} frames | Skipped : {} | Success rate: {:.1f} %".format(ch[1].shape[0], skipped_sweeps, ratio))
print('[INFO] Imported {:.3f} s of data as {} frames of {:.3f} s'.format(total_duration, ch[1].shape[0], T))
print('[INFO] Import done in {:.3f} s ({:.1f} s/s)'.format(t1-t0, total_duration/(t1-t0)))
print()
print("[INFO] Saving to: {}".format(PATH_OUTPUTS))
print("[INFO] Saving period: {:.3f} s ({} plots to save)".format(saving_period, nb_plot_to_save))

print()
print("[INFO] PROCESSING -----------------------")

t, f, d, angles = postprocessing.create_bases(ch, s)
angle_mask = ~(np.isnan(angles) + (np.abs(angles) > s['angle_limit']))
angles_masked = angles[angle_mask]
print("Max ADC range: {:.1f} m".format(d[-1]))

# Create a window
w = np.kaiser(len(ch[1][0]), kaiser_beta)
w *= len(w)/np.sum(w)

# III. Post-processing
timing = dict()  # [DEBUG]
if 1:
    if subtract_background:  # [bool]
        ch[1] = postprocessing.subtract_background(ch[1], w, ch)
        ch[2] = postprocessing.subtract_background(ch[2], w, ch)
    elif subtract_clutter:  # [int]
        ch[1] = postprocessing.subtract_clutter(ch[1], w, ch, subtract_clutter)
        ch[2] = postprocessing.subtract_clutter(ch[2], w, ch, subtract_clutter)
    else:
        ch[1] = w*ch[1]
        ch[2] = w*ch[2]
    
    # III.3. Apply FFTs to channel data to get angle plots
    # [DEBUG] Timing add on
    operations = ['fft', 'concatenate', 'fftshift', 'normalize', 'clim', 'fx', 'fxm', 'log']
    for entry in operations:  # Initialize the timing dict
        timing[entry] = 0
    timing['breakpoint'] = []
    # Set up recurring parameters

    angle_window= np.kaiser(s['angle_pad'], 150)
    #angle_window = np.kaiser(len(angles), 150)
    #assert np.array_equal(angle_window, angle_window_2)

    clim = None
    if 0:
        coefs = [0.008161818583356717,
                 -0.34386493885120994,
                 0.65613506114879,
                 -0.34386493885120994,
                 0.008161818583356717]
    else:
        coefs = [1]
    counter = 0
    ylim_if_time_domain = [MIN_IF_VOLTAGE, MAX_IF_VOLTAGE]
    for index, plot_i in enumerate(tqdm.tqdm(range(len(ch[1])))):
        if plot_i in ch['skipped_sweeps']:  # Skip the processing of zero frames
            assert np.array_equal(ch[1][plot_i], np.zeros(ch[1][plot_i].shape, dtype=np.int16))
            continue
        fxm = None
        counter += 1

        timing['breakpoint'].append(time.perf_counter())  # t0
        a = np.fft.rfft(ch[1][plot_i])
        b = np.fft.rfft(ch[2][plot_i])
        # b *= np.exp(-1j*2*np.pi*channel_dl/(s['c']/(s['f0']+s['bw']/2)))
        b *= np.exp(-1j*2*np.pi*channel_offset*np.pi/180)
        timing['breakpoint'].append(time.perf_counter())  # t1

        if swap_chs:
            x = np.concatenate((b, a)).reshape(2, -1)
        else:
            x = np.concatenate((a, b)).reshape(2, -1)
        timing['breakpoint'].append(time.perf_counter())  # t2

        fx = np.fft.fftshift(np.fft.fft(x, axis=0, n=s['angle_pad']), axes=0)
        timing['breakpoint'].append(time.perf_counter())  # t3

        fx = postprocessing.r4_normalize(fx, d)
        timing['breakpoint'].append(time.perf_counter())  # t4

        if clim is None:
            max_range_i = np.searchsorted(d, max_range)
            clim = np.max(20*np.log10(np.abs(fx[:max_range_i, :]))) + 10
        timing['breakpoint'].append(time.perf_counter())  # t5
        if fxm is None:
            fx = coefs[0]*fx
        else:
            fx += coefs[k]*fx
        timing['breakpoint'].append(time.perf_counter())  # t6

        if flag_Hanning:  # Apply a Hanning window to the peak
            result = []
            center = fx.shape[0] / 2
            for freq in np.transpose(fx):  # Transpose for convenience
                m = np.argmax(np.abs(freq))  # Find the index of the max
                window = np.roll(angle_window, int(round(-center - m)))  # Center window on max
                freq *= window  # Apply the window
                result.append(freq)
            fx = np.transpose(np.stack(result))  # Get back to original shape

        timing['breakpoint'].append(time.perf_counter())  # t7
        fx = fx[angle_mask]
        fxdb = 20*np.log10(np.abs(fx))
        timing['breakpoint'].append(time.perf_counter())  # t8

        # [DEBUG] Update timing 
        for i, entry in enumerate(operations):
            timing[entry] += timing['breakpoint'][i+1] - timing['breakpoint'][i]
        timing['breakpoint'] = []
        
        # Save some of the plots
        if index in index_to_save:
            time_stamp = (PATH_OUTPUTS, T*index, int(T*index), int(1000*(T*index-int(T*index))))

            display.plot_angle(d, fxdb, angles_masked, clim, max_range, time_stamp, show_plot=False)
            display.plot_if_time_domain(t, ch, plot_i, s['fir_gain'], s['adc_bits'], ylim_if_time_domain, time_stamp)

    # [DEBUG] Update timing 
    total_time = sum([v for v in timing.values() if type(v) == float])
    print("\n[INFO] {} iterations in {} s".format(counter, total_time))
    for key in operations:
        print("Operation: {} | Took: {:.6f} s | {:.1f}% of total".format(key, timing[key], 100*timing[key]/total_time))

# III.4. IF spectrum plot
if 1:
    """[IF spectrum plot]"""
    save_path = os.path.join(PATH_OUTPUTS, 'IF_spectrum.png')
    display.plot_if_spectrum(d, ch, plot_i, w, s['fir_gain'], s['adc_bits'], save_path)

# III.4. Range time plot
if 1:
    sweeps = ch[2]  # Using only ch[2] for that ?
    sweep_length = ch[1].shape[1]  # Length of the sweeps
    nb_sweeps = len(sweeps)  # Number of sweeps
    fourier_len = sweep_length/2

    """[TBR] Potentially subtract the background & all"""
    subtract_background = False
    subtract_clutter = False

    if subtract_background:
        background = []
        for i in range(sweep_length):
            x = 0
            for j in range(len(sweeps)):
                x += sweeps[j][i]
            background.append(x/len(sweeps))

    max_range_index = int((4*s['bw']*fourier_len*max_range)/(s['c']*s['if_amplifier_bandwidth']*s['t_sweep']))
    max_range_index = min(max_range_index, sweep_length//2)
    print("Max range index:", max_range_index)
    im = np.zeros((max_range_index-2, nb_sweeps))
    w = np.kaiser(sweep_length, kaiser_beta)
    m = 0

    for e in range(len(sweeps)):
        sw = sweeps[e]
        if subtract_clutter and e > 0:
            sw = [sw[i] - sweeps[e-1][i] for i in range(sweep_length)]
        if subtract_background:
            sw = [sw[i] - background[i] for i in range(sweep_length)]

        sw = [sw[i]*w[i] for i in range(len(w))]  # Take a Kaiser window of the sweep
        fy = np.fft.rfft(sw)[3:max_range_index+1]  # FFT of the sweep
        fy = 20*np.log10((s['adc_ref']/(2**(s['adc_bits']-1)*s['fir_gain']*max_range_index))*np.abs(fy))
        fy = np.clip(fy, -100, float('inf'))
        m = max(m, max(fy))  # Track max value for m
        im[:,e] = np.array(fy)

    if 1:
        f = s['if_amplifier_bandwidth']/2.0
        t = np.linspace(0, overall_decimate*nb_sweeps*(s['t_sweep']+s['t_delay']), im.shape[1])
        xx, yy = np.meshgrid(
            t,
            np.linspace(0,
                        s['c']*max_range_index*s['if_amplifier_bandwidth']/(2*sweep_length)/(s['bw']/s['t_sweep']), im.shape[0]))
        save_path = os.path.join(PATH_OUTPUTS, 'Range-time.png')
        
        """[Range time plot]"""
        display.plot_range_time(t, (xx,yy,im), m, save_path, show_plot=True)