import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import mne

# --- Path Setup ---
results_dir = os.path.join("ds006761", "derivatives", "decoding_results")
target_names = ['self', 'other', 'selfp', 'otherp']

# Find all files
files = [f for f in os.listdir(results_dir) if f.endswith('.h5')]
n_subs = len(files)

# Initialize storage
# We'll use the first file to get dimensions
with h5py.File(os.path.join(results_dir, files[0]), 'r') as hf:
    times = hf['times'][:]
    ch_names = [n.decode('utf-8') for n in hf['ch_names'][:]]
    n_times = len(times)
    n_chans = len(ch_names)

# Shape: [Subjects, Targets, Channels, Times]
all_sl = np.zeros((n_subs, len(target_names), n_chans, n_times))
# Shape: [Subjects, Targets, Times]
all_temp = np.zeros((n_subs, len(target_names), n_times))

# --- Load Data ---
print(f"Loading {n_subs} files...")
for i, f in enumerate(files):
    with h5py.File(os.path.join(results_dir, f), 'r') as hf:
        for t_idx, target in enumerate(target_names):
            if target in hf:
                all_temp[i, t_idx, :] = hf[f'{target}/samples'][:]
                all_sl[i, t_idx, :, :] = hf[f'{target}/searchlight'][:]

# --- 1. Temporal Decoding Plot ---
# Average across subjects
avg_temp = all_temp.mean(axis=0)

plt.figure(figsize=(10, 5))
for t_idx, target in enumerate(target_names):
    plt.plot(times, avg_temp[t_idx], label=target)

plt.axhline(0.33, color='k', linestyle='--', label='Chance (1/3)')  # Assuming 3-choice RPS
plt.axvline(0, color='gray', alpha=0.5)
plt.axvline(1.8, color='gray', alpha=0.5, linestyle=':')  # Part B Start
plt.axvline(3.8, color='gray', alpha=0.5, linestyle=':')  # Part C Start
plt.title("Group Level Temporal Decoding")
plt.xlabel("Time (s)")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# --- 2. Searchlight Spatial Plot (at a specific Time of Interest) ---
# Pick a time point (e.g., 2.5 seconds)
toi = 2.5
time_idx = np.argmin(np.abs(times - toi))

# Average searchlight across subjects for the 'self' target
self_sl_avg = all_sl[:, 0, :, time_idx].mean(axis=0)

# Create MNE Evoked object to use their topomap plotting
info = mne.create_info(ch_names=ch_names, sfreq=4.0, ch_types='eeg')
# You must set a montage for the plot to know sensor locations
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)

evoked = mne.EvokedArray(self_sl_avg[:, np.newaxis], info, tmin=toi)
evoked.plot_topomap(times=toi,
                    size=3,
                    colorbar=True,
                    units='Accuracy',
                    vlim=(0.33, 0.45),
                    title=f"Searchlight: 'Self' at {toi}s")
