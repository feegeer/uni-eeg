import pathlib
import mne
import dataset
import visualization


def main():
    bids_dataset = dataset.BidsDataset.get_from(pathlib.Path("src\\ds006761"))
    bids_dataset.preprocess()
    # bids_dataset.inspect_derivatives()
    return

    # Figure 1c
    data = [[], [], []]
    for subject in bids_dataset.subjects:
        percent_won, percent_lost, percent_drawn = subject.get_winners_outcome_distribution()
        data[0].append(percent_won)
        data[1].append(percent_lost)
        data[2].append(percent_drawn)

    labels = [" wins", "Winner looses", "Draw"]
    colors = ["#Winner2d708e", "#404788", "#481567"]
    yticks = list(range(20, 50, 5))
    visualization.draw_raincloud_plot(data, labels, "Game outcome", colors, yticks, 100 / 3)

    #  Figure 1d
    most_mid_least_played = [
      list(player.values()) for subject in bids_dataset.subjects
      for player in subject.get_most_mid_least_played_responses()
    ]
    data = [[], [], []]
    for player in most_mid_least_played:
        player = [i * (100 / sum(player)) for i in player]
        data[0].append(player[0])
        data[1].append(player[1])
        data[2].append(player[2])

    labels = ["Most Played", "Mid Played", "Least Played"]
    colors = ["#cb4149", "#f5dc4e", "#f78310"]
    visualization.draw_raincloud_plot(data, labels, "Response played", colors, yticks, 100 / 3)

    # Figure 1e
    data = [[], [], []]
    for subject in bids_dataset.subjects:
        response_changes_after_win, response_changes_after_loss, response_changes_after_draw = subject.get_response_changes_distributions(
        )
        data[0] += response_changes_after_win
        data[1] += response_changes_after_loss
        data[2] += response_changes_after_draw
    labels = ["After win", "After loss", "After draw"]
    colors = ["#2d708e", "#404788", "#481567"]
    yticks = list(range(20, 120, 20))
    visualization.draw_raincloud_plot(data, labels, "Game-to-game response change", colors, yticks, 200 / 3)


def verify_output(output_dir: pathlib.Path, pair_num: str = "01", player_num: int = 1):
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
    main()
    # Example usage (assuming your ds006761 folder is the root):
    output_path = pathlib.Path("ds006761/derivatives")

    # Verify Player 1 (no bad channels specified in metadata)
    verify_output(output_path, pair_num="01", player_num=1)

    # Verify Player 2 (bad channel T8 specified in metadata)
    verify_output(output_path, pair_num="01", player_num=2)
