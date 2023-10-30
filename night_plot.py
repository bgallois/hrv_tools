# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np
import hrvanalysis
import sys
import matplotlib.pyplot as plt
import os
import datetime
import time
plt.style.use('seaborn-v0_8-darkgrid')

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def process(path, verbose=True):
    # Load and resample data
    sensor_rate = 100
    resample_rate = 500
    data = pd.read_csv(path, header=None, names=["red", "ir"])
    data.red = pd.to_numeric(data.red, errors="coerce")
    data["time"] = np.arange(0, len(data.red.values)*1/sensor_rate, 1/sensor_rate)
    data = data.dropna().astype(float)
    data = data.reset_index(drop=True)
    data = data.set_index(pd.TimedeltaIndex(data["time"], "s"))
    data = data.resample(pd.Timedelta(1000/resample_rate, "millis")).interpolate("quadratic")


    # Filter data by first applying a low pass filter to remove large wave variation,
    # then a Wiener filter to remove noise
    # and finally a Savgol filter to smooth the peaks.
    sos = scipy.signal.butter(10, 0.01, 'hp', fs=resample_rate, output='sos')
    #data.red = scipy.signal.sosfilt(sos, data.red)
    data.red = scipy.signal.detrend(data.red)
    data.red = scipy.signal.wiener(data.red, int(40*1e-3*resample_rate)) # 40ms window
    data.red = scipy.signal.savgol_filter(data.red, int(80*1e-3*resample_rate), 2) # 20ms window

    # Peaks detection control
    distance = (600*resample_rate)/1000 # Maximum 100 beats per seconds 60*1e3/100
    prominence = 50 # Signal dependant
    peaks, _ = scipy.signal.find_peaks(data.red, distance=distance, prominence=prominence)
    if verbose:
        plt.figure()
        plt.plot(data.time.values/3600, data.red)
        plt.plot(data.time.values[peaks]/3600, data.red[peaks], "x")
        plt.ylabel("Signal")
        plt.xlabel("Time (hours)")
        plt.show()

    hrv_window = 3 # hrv window in minutes
    interval_len = int((60*hrv_window) // (data.time.diff().mean()))
    confidence = 0.01 # Windows with more than confidence*100 % of artefact will be removed

    y = []
    x = []
    for i in range(0, len(data.time.values), interval_len):
        try:
            peaks, _ = scipy.signal.find_peaks(data.red.values[i:i+interval_len], distance=distance, prominence=prominence)
            rr_intervals = np.diff(data.time[peaks]*1e3)
            rr_intervals = hrvanalysis.remove_outliers(rr_intervals=rr_intervals, low_rri=300, high_rri=2000, verbose=False)
            rr_intervals = hrvanalysis.interpolate_nan_values(rr_intervals=rr_intervals,interpolation_method="linear")
            nn_intervals = hrvanalysis.remove_ectopic_beats(rr_intervals=rr_intervals, method="kamath")
            deleted_beat = np.count_nonzero(np.isnan(nn_intervals))
            nn_intervals = hrvanalysis.interpolate_nan_values(rr_intervals=nn_intervals)
            time_domain_features = hrvanalysis.get_time_domain_features(nn_intervals)
            time_domain_features.update(hrvanalysis.get_frequency_domain_features(nn_intervals))
            if not np.isnan(time_domain_features["mean_hr"]) and deleted_beat/len(nn_intervals) < 0.1:
                y.append(time_domain_features)
                x.append(data.time.values[i]/3600)
        except:
            print("Deleted")

    trend_step = 5
    vals = ["sdnn", "pnni_50", "rmssd", "lf", "hf"]
    if verbose:
        fig, axs = plt.subplots(6, 1)
        axs[0].plot(x[trend_step//2:-trend_step//2+1], moving_average([i["mean_hr"] for i in y], trend_step), color="magenta", alpha=.4, label="Trend")
        axs[0].errorbar(x, [i["mean_hr"] for i in y], yerr=[i["std_hr"] for i in y], label="mean_hr", color="C0")
        axs[0].set_ylabel("HR (bpm)")
        for j, k in enumerate(vals):
            axs[j+1].plot(x[trend_step//2:-trend_step//2+1], moving_average([i[k] for i in y], trend_step), color="magenta", alpha=.4)
            axs[j+1].plot(x, [i[k] for i in y], ".-", label=k, color="C{}".format(j+1))
            if k in ["lf", "hf"]:
                axs[j+1].set_ylabel("$ms^2$")
            else:
                axs[j+1].set_ylabel("$ms$")
        axs[-1].set_xlabel("Time (hour)")
        fig.legend()
        plt.show()
    return (x, y)

res = []
date_ = []
is_verbose = len(sys.argv) == 2
for i in sorted(sys.argv[1:]):
    try:
        _, y = process(i, is_verbose)
        res.append(y)
        date_.append(datetime.datetime.strptime(os.path.splitext(i)[0], "%Y%m%d-%H%M%S"))
    except Exception as e:
        print(e, i)

vals = ["mean_hr", "sdnn", "pnni_50", "rmssd", "lf", "hf"]
fig, axs = plt.subplots(6, 1)
for l, k in enumerate(vals):
    axs[l].errorbar(date_, [np.mean([j[k] for j in i]) for i in res], yerr=np.asarray([np.std([j[k] for j in i]) for i in res])/np.sqrt(np.asarray([len([j[k] for j in i]) for i in res])), fmt="-", color="C{}".format(l), label=k)
fig.legend()
plt.show()
