import os
import pathlib
import numpy as np
import pandas as pd
import mne
from scipy.io import loadmat, savemat
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from mne.decoding import SlidingEstimator, cross_val_multiscore
"""
note: index values are in matlab format
Loop through each pair:
    - load the behaviour of the pair --> events tsv file (with the responses)
    - take player 1 reponse (columns 5), player 2 response (column 7), outcome (column 9)
    - create new table for player 1 and another for player 2 such that each table has the following format:
        [player's response | opponent's response | outcome | previous player's response | opponent's previous response]
        note: make sure the player and opponent are relative to whom the table belongs to
    Loop over the 2 players in the pair:
        - load pre-processed (derivatives) EEG data for current player
        - separate epoch into 3 parts: A = [-0.2 to 2.0], B = [1.8 to 4], and C = [3.8 to 5]
        - shift time labes for part B and C to make 0 the start of the reponse (B) or start of the feedback (C)
        - baseline-correction: use the [-0.2 0] as a baseline. Run the baseline corrections for the trial parts
        - remove the first trial of each block (from events and eeg) --> no previous trial
        - average the data into time bins, and re-combine into 1 dataset (rather than 3 parts)
            time windows: A-[0:0.25:1.75;0.25:0.25:2], B-[0:0.25:1.75;0.25:0.25:2], C-[0:0.25:0.75;0.25:0.25:1] 
        Loop over trials:
            Loop over the time bins for this part
                - get the data for time-points in this time bin and average
            - add the data to the the big matrix (one that includes all the data)   
        - convert the data from mne to whatever library has LDA  
        - Loop over things we want to decode
            1 = played self, 2 = played other, 3 = played self previous trial, 4 = played other previous trial

"""

# --- Parameters ---
path_to_data = pathlib.Path("ds006761")
pair_ids = [i for i in range(1, 35) if i not in [10, 23, 24]]
num_trials = 480
num_chan = 64
SFREQ_BINNED = 4  # 4Hz = 1 sample every 250ms


def load_pair_behaviour(pair):
    tsv_path = os.path.join(path_to_data, f"sub-{pair:02d}", "eeg", f"sub-{pair:02d}_task-RPS_events.tsv")
    events_df = pd.read_csv(tsv_path, sep="\t")

    # 1. Perspective Flip for Player 2 Outcomes
    # 1=Draw, 2=P1 Wins, 3=P2 Wins. For P2: 2 becomes 'Loss' (3) and 3 becomes 'Win' (2)
    p2_outcome = events_df['outcome'].replace({2: 3, 3: 2})

    # 2. Vectorized construction for Player 1
    p1_df = pd.DataFrame({
      "this_player_response": events_df['player1_resp'],
      "opponent_player_response": events_df['player2_resp'],
      "outcome": events_df['outcome'],
      # .shift(1) moves the whole column down to get "previous" trial
      "this_player_previous_response": events_df['player1_resp'].shift(1, fill_value=-1),
      "opponent_player_previous_response": events_df['player2_resp'].shift(1, fill_value=-1)
    })

    # 3. Vectorized construction for Player 2
    p2_df = pd.DataFrame({
      "this_player_response": events_df['player2_resp'],
      "opponent_player_response": events_df['player1_resp'],
      "outcome": p2_outcome,
      "this_player_previous_response": events_df['player2_resp'].shift(1, fill_value=-1),
      "opponent_player_previous_response": events_df['player1_resp'].shift(1, fill_value=-1)
    })

    # 4. Critical: Reset previous trial data at block boundaries (every 40 trials)
    # Without this, trial 41 would think trial 40 (end of previous block) is its history.
    block_starts = np.arange(0, len(events_df), 40)
    for df in [p1_df, p2_df]:
        df.loc[block_starts, "this_player_previous_response"] = -1
        df.loc[block_starts, "opponent_player_previous_response"] = -1

    return p1_df, p2_df


# --- Loop Over Pairs ---
for pair in pair_ids:
    print(f"Loading pair {pair}")
    p1_events, p2_events = load_pair_behaviour(pair)

    for ppt in [1, 2]:
        print(f"   ppt {ppt}")

        # Perspective shift for behavioral data
        if ppt == 1:
            behav = np.column_stack([raw_behav, np.full((num_trials, 2), np.nan)])
            behav[1:, 3:] = raw_behav[:-1, :2]  # Previous trial
        else:
            behav = np.zeros((num_trials, 5))
            behav[:, [0, 1]] = raw_behav[:, [1, 0]]  # Swap self/other
            # Adjust outcome relative to player 2
            p1_out = raw_behav[:, 2]
            behav[p1_out == 1, 2] = 1  # Draw
            behav[p1_out == 2, 2] = 3  # P1 win -> P2 lose
            behav[p1_out == 3, 2] = 2  # P1 lose -> P2 win
            behav[1:, 3:] = behav[:-1, :2]  # Previous trial

        fif_path = f"{path_to_data}/derivates/pair-{pair:02d}_player-{ppt:01d}_task-RPS_eeg.fif"
        epochs = mne.read_epochs(fif_path, preload=True)

        # --- Preprocessing & Windowing ---
        # 1. Re-reference to Average
        epochs.set_eeg_reference("average")

        # 2. Split and task-specific baseline (MATLAB parts A, B, C)
        # Part A: [0, 2s], B: [2, 4s], C: [4, 5s] relative to trial onset
        ep_a = epochs.copy().crop(tmin=-0.2, tmax=2.0).apply_baseline((-0.2, 0))
        ep_b = epochs.copy().crop(tmin=1.8, tmax=4.0).apply_baseline((1.8, 2.0))
        ep_c = epochs.copy().crop(tmin=3.8, tmax=5.0).apply_baseline((3.8, 4.0))

        # Recombine data [Trials x Channels x Times]
        # We manually concatenate the data arrays to create one task-continuous epoch
        data_comb = np.concatenate([ep_a.get_data(), ep_b.get_data(), ep_c.get_data()], axis=2)
        info = ep_a.info
        epochs_comb = mne.EpochsArray(data_comb, info, tmin=0)

        # 3. Binning: Resample to 4Hz (Each sample = mean of 250ms)
        epochs_comb.resample(SFREQ_BINNED)

        # 4. Remove block starts (indices 0, 40, 80...)
        rem_idx = np.arange(0, 480, 40)
        epochs_comb.drop(rem_idx)
        behav_clean = np.delete(behav, rem_idx, axis=0)

        # --- Decoding Loop ---
        target_names = ['self', 'other', 'selfp', 'otherp']
        target_cols = [0, 1, 3, 4]
        results = {'decoding_accuracy': {}, 'searchlight_acc': {}}

        # Define adjacency for Spatial Searchlight (based on channel positions)
        adjacency, ch_names = mne.channels.find_ch_adjacency(epochs_comb.info, type='eeg')

        for name, col in zip(target_names, target_cols):
            y = behav_clean[:, col]
            valid = ~np.isnan(y) & (y > 0)

            X_valid = epochs_comb.get_data()[valid]
            y_valid = y[valid]

            # Replicate CoSMo SNR averaging
            X_avg, y_avg = cosmo_average_samples(X_valid, y_valid)

            # --- 1. Temporal Decoding (Sliding Estimator) ---
            clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
            time_gen = SlidingEstimator(clf, scoring='accuracy', n_jobs=-1)
            scores = cross_val_multiscore(time_gen, X_avg, y_avg, cv=10)
            results['decoding_accuracy'][name] = scores.mean(axis=0)

            # --- 2. Spatial Searchlight (Iterating through channel neighborhoods) ---
            # To mimic 'cosmo_meeg_chan_neighborhood' with count=4:
            spatial_scores = np.zeros((num_chan, X_avg.shape[2]))  # [Chans x TimeBins]

            for i in range(num_chan):
                # Find the 3 closest neighbors + the channel itself = cluster of 4
                # (using indices from the adjacency matrix)
                neighbors = adjacency[i].indices
                X_spatial = X_avg[:, neighbors, :]

                # Run sliding estimator on this small spatial cluster
                # We mean across the neighbors to create a "virtual channel" or
                # let the LDA handle the 4-feature vector
                scores_spat = cross_val_multiscore(time_gen, X_spatial, y_avg, cv=10)
                spatial_scores[i, :] = scores_spat.mean(axis=0)

            results['searchlight_acc'][name] = spatial_scores

        # Save Results
        save_out = path_to_data / "derivatives" / f"pair-{pair:02d}_player-{ppt:01d}_decoding_results.mat"
        savemat(save_out, results)
        print(f"      Saved: {save_out.name}")
