import sys
import matplotlib.pyplot as plt
import numpy as np
import ast
from scipy.signal import butter, filtfilt
import tqdm
import argparse
import os
import time
import collections
import copy
#import subprocess as sp

# Custom packages
from fmcw import *

fs = 2e6  # IF amplifier bandwidth
fir_gain = 9.0
c = 299792458.0
adc_ref = 1
adc_bits = 12

max_range = 250
d_antenna = 28e-3 #Antenna distance
#channel_dl = 0e-3 #Channel length difference
angle_limit = 55
channel_offset = 21
swap_chs = True
sweeps_to_read = None
angle_pad = 100
decimate_sweeps = 1
kaiser_beta = 6  # Hanning window
MIN_IF_VOLTAGE = -1.1
MAX_IF_VOLTAGE = 1.1



"""
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--log", type=str, help="path to log file", required=True)
args = vars(ap.parse_args())
PATH_LOG = args["log"]
"""
PATH_LOG = 'fmcw3.log'
BINARY = True
ENCODING = 'latin1' if BINARY else None
start = b'\x7f' if BINARY else '\x7f'

if PATH_LOG[0] != '/':
    PATH_LOG = os.path.join(os.getcwd(), PATH_LOG)
PATH_FOLDER_LOG = os.path.split(PATH_LOG)[0]
FOLDER_NAME = os.path.split(PATH_FOLDER_LOG)[1]
PATH_OUTPUTS = os.path.join(PATH_FOLDER_LOG, 'Outputs')
STRIDE = 55

"""[Python 3]
if os.path.isdir(PATH_OUTPUTS):
    remove_output_dir = ['rm', '-r', PATH_OUTPUTS]
    sp.run(remove_output_dir)
create_output_dir = ['mkdir', PATH_OUTPUTS]
sp.run(create_output_dir)
"""
if os.path.isdir(PATH_OUTPUTS):
    os.system('rm -r '+PATH_OUTPUTS)
os.system('mkdir '+PATH_OUTPUTS)




read_mode = 'rb' if BINARY else 'r'
with open(PATH_LOG, read_mode) as f:
    settings = postprocessing.read_settings(f, encoding=ENCODING)
    DURATION = settings['duration']
    f0 = settings['f0']
    bw = settings['bw']
    tsweep = settings['tsweep']
    tdelay = settings['tdelay']
    half = settings['downsampler']
    quarter = settings['quarter']
    decimate = settings['decimate']
    channels = int(settings['a']) + int(settings['b'])
    print("[INFO] {} byte of setting data found: \n{}".format(f.tell(), settings))

    multiplier = 1
    if half:
        fs /= 2
        multiplier = 2
        if quarter:
            fs /= 2
            multiplier = 4

    fc = f0+bw/2
    wl = c/(f0+bw/2)
    samples_in_sweep = int(tsweep*fs)
    NBYTES_SWEEP = channels*samples_in_sweep*multiplier
    print("[INFO] Samples in sweep: {} | Sweep length: {} byte".format(samples_in_sweep, NBYTES_SWEEP))

    if sweeps_to_read != None:
        samples = (2+NBYTES_SWEEP)*sweeps_to_read
    else:
        samples = None

    # A frame starts with a signal, a frame number and then NBYTES_SWEEP byte of data
    current_frame_number = postprocessing.find_start(f, start, NBYTES_SWEEP)
    print('\n[INFO] Found a start signal!')

    t0 = time.perf_counter()
    data, counter_skipped_lines = postprocessing.import_data(f, start, current_frame_number, NBYTES_SWEEP, samples, decimate_sweeps, channels)
    t1 = time.perf_counter()
    #assert 1==0


ch1 = np.array(data[1], dtype=np.int16)
ch2 = np.array(data[2], dtype=np.int16)
# CRITICAL, DO NOT REMOVE THIS ASSERTION
assert ch1.shape == ch2.shape
print("[INFO] Found arrays of shape", ch1.shape)
"""
index = np.random.randint(len(ch1))
plt.figure()
plt.plot(ch1[index])
plt.plot(ch2[index])
plt.title(index)
plt.show()
"""


decimate_sweeps *= decimate
if channels == 2:
    assert ch1.shape == ch2.shape
T = (tsweep+tdelay)*decimate_sweeps
expected_lines = ch1.shape[0]
skipped_lines = counter_skipped_lines
ratio = 100*(1-skipped_lines/expected_lines)
total_duration = T*ch1.shape[0]  # Total recording time

print()
print('[INFO] Done reading {:.3f} s of data in {:.3f} s ({:.1f} FPS)'.format(total_duration, t1-t0, total_duration/(t1-t0)))
print("[INFO] Found {} frames | Skipped : {} | Success rate: {:.1f} %".format(ch1.shape[0], skipped_lines, ratio))

print()
print("[INFO] PROCESSING -----------------------")
print("[INFO] Data period: {:.3f} s".format(T))

t = np.linspace(0, tsweep, len(ch1[0]))
f = np.linspace(0, fs/2, len(ch1[0])//2+1)
d = postprocessing.f_to_d(f, bw, tsweep)
print("Max freq: {} | B: {} | Tc: {}".format(f[-1], bw, tsweep))
print("Max ADC range: {:.1f} m".format(d[-1]))
#assert 1==0
angles = 180/np.pi*np.arcsin(np.linspace(1, -1, angle_pad)*wl/(2*d_antenna))
angle_mask = ~(np.isnan(angles) + (np.abs(angles) > angle_limit))
angles_masked = angles[angle_mask]
plot_i = -12

print("[INFO] Saving to: {}".format(PATH_OUTPUTS))
print("[INFO] Saving period: {:.3f} s".format(STRIDE*T))


if 0:
    #Add 0 dBFs signal for testing purposes
    ch1[plot_i] += fir_gain*2**(adc_bits-1)*np.sin(2*np.pi*50e3*np.linspace(0, tsweep, len(ch1[0])))

w = np.kaiser(len(ch1[0]), kaiser_beta)
w *= len(w)/np.sum(w)
timing = dict()
if 1:
    subtract_background = False
    subtract_clutter = False  # Subtract the previous plot only

    """[TBR] Old method from Henrik
    if subtract_background: # Averaging on the whole dataset. Better not have moved the radar
        background1 = np.zeros(len(ch1[0]))
        background2 = np.zeros(len(ch1[0]))
        for i in range(len(ch1[0])):
            x1 = 0
            x2 = 0
            for j in range(len(ch1)):
                x1 += ch1[j][i]
                x2 += ch2[j][i]
            background1[i] = (x1/len(ch1))
            background2[i] = (x2/len(ch1))
    """
    
    if subtract_background:
        # Construct the background image of the non-skipped sweeps, and subtract it from the channels
        ch1 = postprocessing.subtract_background(ch1, w, data)
        ch2 = postprocessing.subtract_background(ch2, w, data)
    elif subtract_clutter:
        clutter_averaging = 10  # Number of previous sweeps to use for subtraction
        ch1 = postprocessing.subtract_clutter(ch1, w, data, clutter_averaging)
        ch2 = postprocessing.subtract_clutter(ch2, w, data, clutter_averaging)
    else:
        ch1 = w*ch1
        ch2 = w*ch2


    angle_window = np.kaiser(len(angles), 150)
    clim = None

    if 0:
        coefs = [0.008161818583356717,
        -0.34386493885120994,
        0.65613506114879,
        -0.34386493885120994,
        0.008161818583356717]

        moving_average = len(coefs)
    else:
        coefs = [1]
        moving_average = 1

    counter = 0
    timing['fft'] = 0
    timing['concatenate'] = 0
    timing['fftshift'] = 0
    timing['normalize'] = 0
    timing['clim'] = 0
    timing['fxm'] = 0
    timing['fx'] = 0
    timing['log'] = 0
    for index, plot_j in enumerate(tqdm.tqdm(range(moving_average-1, len(ch1)))):
        fxm = None
        for k in range(moving_average):
            """[Old Henrik] Includes hook to verify that my new method is identical
            plot_i = plot_j-moving_average+k+1
            print("[INFO] Processing plot_i", plot_i)
            if subtract_background:
                a = w*(ch1[plot_i] - background1)
                b = w*(ch2[plot_i] - background2)
                assert np.array_equal(a, ch1_2[plot_i])
                assert np.array_equal(b, ch2_2[plot_i])
                print("[INFO] Background verified")
            elif subtract_clutter:  # You cannot subtract both the background and clutter
                if plot_i == 0:
                    a = w*(ch1[plot_i])
                    b = w*(ch2[plot_i])
                    #continue
                else:
                    a = w*(ch1[plot_i] - ch1[plot_i-1])
                    b = w*(ch2[plot_i] - ch2[plot_i-1])
                    assert np.array_equal(a, ch1_2[plot_i])
                    assert np.array_equal(b, ch2_2[plot_i])
                    print("[INFO] Clutter verified")
            else:
                a = w*ch1[plot_i]
                b = w*ch2[plot_i]
                assert np.array_equal(a, ch1_2[plot_i])
                assert np.array_equal(b, ch2_2[plot_i])
                print("[INFO] None verified")
            
            a = np.fft.rfft(a)
            b = np.fft.rfft(b)
            """

            # New processing
            plot_i = plot_j
            counter += 1

            t0 = time.perf_counter()
            #print("ch1 & ch2 are ", ch1[plot_i].shape, ch1[plot_i].shape)
            a = np.fft.rfft(ch1[plot_i])
            b = np.fft.rfft(ch2[plot_i])
            #b *= np.exp(-1j*2*np.pi*channel_dl/(c/(f0+bw/2)))
            b *= np.exp(-1j*2*np.pi*channel_offset*np.pi/180)
            #print("a & b are ", a.shape, b.shape)
            t1 = time.perf_counter()

            if swap_chs:
                x = np.concatenate((b, a)).reshape(2, -1)
            else:
                x = np.concatenate((a, b)).reshape(2, -1)
            t2 = time.perf_counter()
            #print("x", x.shape)

            fx = np.fft.fftshift(np.fft.fft(x, axis=0, n=angle_pad), axes=0)
            #print("fx", fx.shape)
            t3 = time.perf_counter()

            fx = postprocessing.r4_normalize(fx, d)
            t4 = time.perf_counter()

            if clim == None:
                max_range_i = np.searchsorted(d, max_range)
                clim = np.max(20*np.log10(np.abs(fx[:max_range_i,:]))) + 10
            t5 = time.perf_counter()
            if fxm is None:
                fx = coefs[0]*fx
            else:
                fx += coefs[k]*fx
            t6 = time.perf_counter()
        # fx = fxm
        #print("angle_window", angle_window.shape)
        fx_initial = copy.copy(fx)
        if 0:
            """[Old Henrik]"""
            for j in range(fx.shape[1]):
                #plt.figure()
                fj = fx[:,j]
                #plt.plot(fj)
                #assert 1==0
                m = np.argmax(np.abs(fj))  # Find the index of the largest absolute value across all frequencies?
                window = np.roll(angle_window, int(round(-fx.shape[0]/2 - m)))
                fx[:,j] *= window

                #plt.plot(fx[:,j])
                #plt.plot(np.max(fx)*window)
                #plt.legend(('fj', 'fx', 'window'))
                #plt.show()
            """"""
            """[My take]
            # Alternative option
            result = []
            center = fx.shape[0] / 2
            for index, freq in enumerate(np.transpose(fx)):
                m_2 = np.argmax(np.abs(freq))
                window_2 = np.roll(angle_window, int(round(-center - m_2)))
                freq *= window_2
                result.append(freq)
            fx_2 = np.transpose(np.stack(result))
            assert np.array_equal(fx, fx_2)
            #print("Valid")
            """
        #assert 1==0


        t7 = time.perf_counter()
        #print(angle_mask)
        fx = fx[angle_mask]
        #print(np.max(np.abs(fx_initial[angle_mask] - fx)))
        fxdb = 20*np.log10(np.abs(fx))
        t8 = time.perf_counter()

        # Update timing
        timing['fft'] += t1 - t0
        timing['concatenate'] += t2 - t1
        timing['fftshift'] += t3 - t2
        timing['normalize'] += t4 - t3
        timing['clim'] += t5 - t4
        timing['fxm'] += t6 - t5
        timing['fx'] += t7 - t6
        timing['log'] += t8 - t7

        """[Usual figure display]"""
        if index%STRIDE==0:
            fig = plt.figure()
            if 0:
                ax = fig.add_subplot(111, polar=True)
                imgplot = ax.pcolormesh(angles_masked*np.pi/180, d, fxdb.transpose())
            elif 0:
                r, t = np.meshgrid(d, angles_masked*np.pi/180)
                x = r*np.cos(t)
                y = -r*np.sin(t)
                imgplot = plt.pcolormesh(x, y, fxdb)
                plt.colorbar()
                ylim = 90*np.sin(angles_masked[0]*np.pi/180)
                #plt.ylim([-ylim, ylim])
                plt.ylim([-30, 30])
                plt.xlim([d[0], max_range])
                plt.ylabel("Cross-range [m]")
            else:
                imgplot = plt.pcolormesh(d, angles_masked, fxdb)
                plt.colorbar()
                plt.ylim([angles_masked[0], angles_masked[-1]])
                plt.xlim([d[0], max_range])
                plt.ylabel("Angle [$^o$]")
            imgplot.set_clim(clim-50,clim)
            plt.title(FOLDER_NAME+'\n{0:.2f} s'.format( plot_i*(tsweep+tdelay)*decimate_sweeps))
            plt.xlabel("Range [m]")
            plt.savefig(os.path.join(PATH_OUTPUTS, 'range_{:04d}.png'.format(plot_i)))
            #plt.show()
            plt.close()
        """"""
        
        """[IF time domain]"""
        if index%STRIDE==0:
            # Plot the IF time-domain values
            plt.figure()
            ch1_data = np.array(ch1[plot_i])/(fir_gain*2**(adc_bits-1))
            ch2_data = np.array(ch2[plot_i])/(fir_gain*2**(adc_bits-1))
            #ch1_data = np.array(ch1[plot_i])/fir_gain
            #ch2_data = np.array(ch2[plot_i])/fir_gain
            plt.plot(t, ch1_data)
            if channels == 2:
                plt.plot(t, ch2_data)
            #plt.title(FOLDER_NAME+'\nIF time-domain at {:.2f} s with fir_gain {} and adc_divider {}'.format(plot_i*(tsweep+tdelay)*decimate_sweeps, fir_gain, 2**(adc_bits-1)))
            plt.title('IF time-domain at {:.2f} s with fir_gain {} and adc_divider {}\nCH1: mean: {:.3f} | std: {:.3f}\nCH2: mean: {:.3f} | std: {:.3f}'.format(plot_i*(tsweep+tdelay)*decimate_sweeps, fir_gain, 2**(adc_bits-1), np.mean(ch1_data), np.std(ch1_data), np.mean(ch2_data), np.std(ch2_data)))
            plt.ylabel("Voltage [V]")
            plt.xlabel("Time [s]")
            plt.grid(True)
            plt.ylim([MIN_IF_VOLTAGE, MAX_IF_VOLTAGE])
            
            plt.legend(["Channel 1", "Channel 2"], loc='upper right')
            plt.savefig(os.path.join(PATH_OUTPUTS, 'IF_{:04d}.png'.format(plot_i)))
            #print("Saved {}".format(plot_i))
            plt.close()
        """"""

    total_time = sum(timing.values())
    print("\n[INFO] {} iterations in {} s".format(counter, total_time))
    for key, value in timing.items():
        print(key, value, value/total_time)
    #print(timing)

if 0:
    plt.figure()
    plt.plot(t, np.array(ch1[plot_i]))
    if channels == 2:
        plt.plot(t, np.array(ch2[plot_i]))
    plt.title(FOLDER_NAME+'\nIF time-domain at {:.3f} s with fir_gain {} and adc_divider {}'.format(120*(tsweep+tdelay)*decimate_sweeps, fir_gain, 2**(adc_bits-1)))
    plt.ylabel("Amplitude [V]")
    plt.xlabel("Time [s]")
    plt.legend(["Channel 1", "Channel 2"], loc='upper right')
    #plt.ylim([MIN_IF_VOLTAGE, MAX_IF_VOLTAGE])
    plt.grid(True)
    #plt.show()
    plt.savefig(os.path.join(PATH_OUTPUTS, 'IF_waveforms.png'))
    plt.close()

if 1:
    x1 = np.array(ch1[plot_i], dtype=np.float)
    x2 = np.array(ch2[plot_i], dtype=np.float)

    x1 *= w/(fir_gain*2**(adc_bits-1))
    x2 *= w/(fir_gain*2**(adc_bits-1))
    fx1 = 2*np.fft.rfft(x1)/(len(x1))
    fx2 = 2*np.fft.rfft(x2)/(len(x2))
    #fx1 = r4_normalize(fx1, d)
    fx1 = 20*np.log10(np.abs(fx1))
    fx2 = 20*np.log10(np.abs(fx2))

    #print np.mean(fx1[40:])
    #print np.mean(fx2[40:])
    plt.figure()
    plt.plot(d, fx1, label='Channel 1')
    plt.plot(d, fx2, label='Channel 2')
    plt.legend(loc='best')
    plt.title('IF spectrum')
    plt.ylabel("Amplitude [dBFs]")
    plt.xlabel("Distance [m]")
    plt.savefig(os.path.join(PATH_OUTPUTS, 'IF_spectrum.png'))
    plt.close()


if 1:
    sweeps = ch2  # Using only ch2 for that ?
    print(sweeps.shape)

    subtract_background = False
    subtract_clutter = False
    

    #sweep_length = len(ch1[0])
    sweep_length = ch1.shape[1] # Length of the sweeps

    # Create the background image inefficiently
    if subtract_background:
        background = []
        for i in range(sweep_length):
            x = 0
            for j in range(len(sweeps)):
                x += sweeps[j][i]
            background.append(x/len(sweeps))

        
    nb_sweeps = len(sweeps)  # Number of sweeps 

    #print nb_sweeps, "nb_sweeps"
    #fourier_len_old = len(sweeps[0])/2
    fourier_len = sweep_length/2
    #assert fourier_len_old == fourier_len

    max_range_index = int((4*bw*fourier_len*max_range)/(c*fs*tsweep))
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
        fy = 20*np.log10((adc_ref/(2**(adc_bits-1)*fir_gain*max_range_index))*np.abs(fy))
        fy = np.clip(fy, -100, float('inf'))
        m = max(m, max(fy))  # Track max value for m
        im[:,e] = np.array(fy)

    if 1:
        f = fs/2.0

        t = np.linspace(0, decimate_sweeps*nb_sweeps*(tsweep+tdelay), im.shape[1])
        xx, yy = np.meshgrid(
            #np.linspace(0,im.shape[1]-1, im.shape[1]),
            t,
            np.linspace(0, c*max_range_index*fs/(2*sweep_length)/((bw/tsweep)), im.shape[0]))

        plt.figure()
        plt.ylabel("Range [m]")
        plt.xlabel("Time [s]")
        plt.xlim([t[0], t[-1]])
        plt.title(FOLDER_NAME+'\nRange-time plot')
        imgplot = plt.pcolormesh(xx,yy,im)
        imgplot.set_clim(m-80,m)
        plt.colorbar()
        plt.savefig(os.path.join(PATH_OUTPUTS, 'Range-time.png'), dpi=500)
        plt.show()
        #Save png of the plot
        #image.imsave('range_time_raw.png', np.flipud(im))
        
        plt.close()
        #plt.savefig('range_time.png', dpi=500)

plt.show()
