from scipy.signal import savgol_filter
from merge_tracking_segments import count
from config import resultfolder
from pathlib import Path
import pandas as pd
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

framerate = 30
files = sorted(resultfolder.rglob("*.paramecia_merged.csv"))

def is_wt(name: Path):
    return 'WT' in str(name)
    
results = {'WT': [], 'Mecp2': []}

for f in files:
    param_tracking =  pd.read_csv(f)
    frames, num_param = count(param_tracking)
    num_param_smooth = savgol_filter(num_param, window_length=1800, polyorder=2)
    if is_wt(f):
        results['WT'].append(num_param_smooth)
    else:
        results['Mecp2'].append(num_param_smooth)

cutoff = np.min([len(x) for x in results['WT']] + [len(x) for x in results['Mecp2']])

wt = np.vstack([-(arr[0:cutoff] - arr[0]) for arr in results['WT']])
mecp2 = np.vstack([-(arr[0:cutoff] - arr[0]) for arr in results['Mecp2']])
frames = np.arange(cutoff)*1/framerate

def plot(frames, results: NDArray, col):
    m = np.mean(results, axis = 0)
    std = np.std(results, axis = 0)
    handle, = plt.plot(frames, m, color=col)
    plt.fill_between(frames, m+std, m-std, color=col, alpha=0.2)
    plt.xlabel('time (sec)')
    plt.ylabel('# param')
    return handle

p0 = plot(frames, wt, 'b')
p1 = plot(frames, mecp2, 'r')
plt.legend([p0, p1], ['WT','Mecp2'])
plt.savefig('52lux')
plt.show()
