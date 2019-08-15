# f0 [Hz] | Starting frequency of the chirp
# bw [Hz] | Bandwidth used for the chirp
# t_sweep [s] | Duration of the chirp/sweep
# t_delay [s] | Delay between two chirps/sweeps
# pa_off_advance [s] | NOT SURE YET
# acquisition_decimate [-] | Sweeps to skip. 0 means no sweeps are skipped
# down_sampler [bool] | MANDATORY. Divides the ADC clock by 2. TO DO: fix bug when False
# quarter [bool] | Divides the ADC clock by 4
# 'byte_usb_read': 0x10000,
# 'range_time_to_display': 1,
# 'max_range': 50,
# 'channel_offset': 21,
# 'swap_chs': True,
# 'sweeps_to_read': None,
# 'soft_decimate': 0,
# 'kaiser_beta': 6,
# 'subtract_background': False,
# 'subtract_clutter': 0,
# 'flag_Hanning': False,
# 'real_time_recall': duration,
# 'encoding': encoding,
# 'timeout': 0.5,
# 'path_raw_log': os.path.join(path_log_folder, 'fmcw3.log'),
# 'path_csv_log': os.path.join(path_log_folder, 'fmcw3.csv'),
# 'refresh_period': 0.25,
