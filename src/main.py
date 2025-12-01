import pathlib
import mne
import subject


def main():
    bids_dataset = subject.BidsDataset.get_from(
        pathlib.Path(r"C:\Users\poten\uni-eeg\src\ds006761")
    )
    # bids_dataset = subject.BidsDataset.get_from(pathlib.Path("ds006761"))
    bids_dataset.preprocess()

def verify_output(output_dir, pair_num="01", player_num=1):
    """Loads and prints key information for a processed Epochs file."""
    
    # Construct the expected filename
    fname = output_dir / f"pair-{pair_num}_player-{player_num}_task-RPS_eeg_epo.fif"
    
    if not fname.exists():
        print(f"Verification FAILED: File {fname.name} not found.")
        return

    # Load the Epochs file
    epochs = mne.read_epochs(fname, preload=False, verbose=False) 
    
    print(f"\n### Verification for {fname.name} ###")
    print(f"Total Epochs (Trials): {len(epochs)}")
    print(f"Time Window (tmin/tmax): {epochs.times[[0, -1]]} seconds")
    print(f"Sampling Rate (Hz): {epochs.info['sfreq']}")
    print(f"Number of Channels: {len(epochs.ch_names)}")
    print(f"Sample Channel Names (10-20 system): {epochs.ch_names[0:5]}")
    
    # Verify processing steps were applied
    if epochs.info['bads']:
        print(f"Bad Channels in info['bads']: {epochs.info['bads']}")
    else:
        print("Bad Channels: None")
    
    if epochs.info['dig']:
        print("Montage Check: info['dig'] is POPULATED.")
    else:
        print("Montage Check: info['dig'] is EMPTY (FAIL).")

if __name__ == "__main__":
    # main()
    # Example usage (assuming your ds006761 folder is the root):
    output_path = pathlib.Path("C:/Users/poten/uni-eeg/src/ds006761/derivatives")

    # Verify Player 1 (no bad channels specified in metadata)
    verify_output(output_path, pair_num="01", player_num=1)

    # Verify Player 2 (bad channel T8 specified in metadata)
    verify_output(output_path, pair_num="01", player_num=2)
