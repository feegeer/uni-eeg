import collections
from collections.abc import Mapping
import csv
import dataclasses
import enum
import gc
import pathlib
from typing import Any

import mne
import mne_bids
import scipy.io

# --- Global Constants & Setup (Mimicking FieldTrip Layout/Geometry Loading) ---
# The standard BioSemi codes in order (A1-A32 then B1-B32) corresponding
# to the channel sequence used by FieldTrip's biosemi64.lay template.
BIOSEMI_ORDERED_CODES = [
  'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18',
  'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30', 'A31', 'A32', 'B1', 'B2', 'B3',
  'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20',
  'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B28', 'B29', 'B30', 'B31', 'B32'
]

# Get the standard 10-20 names from MNE's built-in montage (64 channels)
montage_1020_ref = mne.channels.make_standard_montage('standard_1020')
TEN_TWENTY_ORDERED_LABELS = montage_1020_ref.ch_names[:64]

# CRITICAL MAPPING: BioSemi Code -> 10-20 Label.
# COMPARISON (MATLAB): This is equivalent to loading the labels from 'biosemi64.lay'.
BIOSEMI_CODE_TO_1020_LABEL = dict(zip(BIOSEMI_ORDERED_CODES, TEN_TWENTY_ORDERED_LABELS))

# 1. Load the 3D Coordinates from the .mat file
# CRITICAL: Replace 'path/to/biosemi64.mat' with the actual path to your file.
# COMPARISON (MATLAB): Equivalent to 'load('biosemi64.mat');'
try:
    mat_contents = scipy.io.loadmat("src/biosemi64.mat")
    # Assuming the coordinate array is stored under the key 'biosemi64' inside the .mat file.
    biosemi_coords_3d = mat_contents['biosemi64']
except FileNotFoundError:
    print("FATAL ERROR: biosemi64.mat not found. Please update the file path in the script.")
    raise

# 2. Create a custom MNE montage from the .mat data.
# Note: Assuming the data is in centimeters (cm) and converting to meters (m) as MNE expects meters.
# If your data is already in meters, remove the division by 100.
ch_pos_dict_3d = dict(zip(TEN_TWENTY_ORDERED_LABELS, biosemi_coords_3d / 14))
FULL_MNE_BISEOMI_MONTAGE = mne.channels.make_dig_montage(ch_pos=ch_pos_dict_3d, coord_frame='head')
# COMPARISON (MATLAB): This MNE object now holds the 3D positions needed for ft_channelrepair.

# --- Data Structures (Enums and Dataclasses) ---


class Gender(enum.Enum):
    MALE = "M"
    FEMALE = "F"
    NON_BINARY = "NB"


class Handedness(enum.Enum):
    LEFT = "L"
    RIGHT = "R"
    AMBIDEXTROUS = "A"


class Outcome(enum.Enum):
    DRAW = 1
    PLAYER1_WINS = 2
    PLAYER2_WINS = 3


class Response(enum.Enum):
    NO_RESPONSE = 0
    ROCK = 1
    PAPER = 2
    SCISSORS = 3


@dataclasses.dataclass
class Event:
    onset: float
    duration: float
    onset_sample: int
    trial_num: int
    player1_response: Response
    player2_response: Response
    player1_rt: float
    player2_rt: float
    outcome: Outcome


@dataclasses.dataclass
class Player:
    gender: Gender
    age: int
    handedness: Handedness
    preprocessing_channels_fixed: list[str]


@dataclasses.dataclass
class Subject:
    id: str
    player1: Player
    player2: Player
    events: list[Event]

    @staticmethod
    def from_tsv_row(bids_root, row: Mapping[Any]) -> "Subject":
        pid = row["participant_id"]
        # --- Parsing Player Metadata ---
        p1_bad = [] if not row["player1_pre_processing_channels_fixed"] else [
          ch.strip() for ch in row["player1_pre_processing_channels_fixed"].split(",") if ch.strip()
        ]
        player1 = Player(Gender(row["player1_gender"]), int(row["player1_age"]), Handedness(row["player1_handedness"]),
                         p1_bad)
        p2_bad = [] if not row["player2_pre_processing_channels_fixed"] else [
          ch.strip() for ch in row["player2_pre_processing_channels_fixed"].split(",") if ch.strip()
        ]
        player2 = Player(Gender(row["player2_gender"]), int(row["player2_age"]), Handedness(row["player2_handedness"]),
                         p2_bad)
        events = get_events_for_subject(bids_root, pid)
        return Subject(pid, player1, player2, events)

    def get_overall_outcome(self) -> Outcome:
        player1_wins = sum(1 for event in self.events if event.outcome == Outcome.PLAYER1_WINS)
        player2_wins = sum(1 for event in self.events if event.outcome == Outcome.PLAYER2_WINS)

        if player1_wins == player2_wins:
            return Outcome.DRAW

        return Outcome.PLAYER1_WINS if player1_wins > player2_wins else Outcome.PLAYER2_WINS

    def get_winners_outcome_distribution(self) -> tuple[float, float, float]:
        outcomes = [
          event.outcome for event in self.events
          if not (event.player1_response == Response.NO_RESPONSE or event.player2_response == Response.NO_RESPONSE)
        ]
        overall_outcome = self.get_overall_outcome()
        winner_won = 0
        winner_lost = 0
        drawn = 0
        for outcome in outcomes:
            if outcome == overall_outcome:
                winner_won += 1
            elif outcome == Outcome.DRAW:
                drawn += 1
            else:
                winner_lost += 1

        scale = 100 / len(outcomes)
        return winner_won * scale, winner_lost * scale, drawn * scale

    def get_game_outcomes(self) -> dict[Outcome, int]:
        return collections.Counter(event.outcome for event in self.events)

    def get_most_mid_least_played_responses(self) -> tuple[dict[Outcome, int], dict[Outcome, int]]:
        player1_responses = collections.Counter(
          event.player1_response for event in self.events
          if event.player1_response != Response.NO_RESPONSE and event.player2_response != Response.NO_RESPONSE)
        player2_responses = collections.Counter(
          event.player2_response for event in self.events
          if event.player1_response != Response.NO_RESPONSE and event.player2_response != Response.NO_RESPONSE)
        return dict(player1_responses.most_common()), dict(player2_responses.most_common())

    def get_response_changes_distributions(self) -> tuple[list[float], list[float], list[float]]:
        NUM_BLOCKS = 12
        TRIALS_PER_BLOCK = 40

        p1_win, p1_lose, p1_draw = [], [], []
        p2_win, p2_lose, p2_draw = [], [], []

        for block in range(NUM_BLOCKS):
            block_start = block * TRIALS_PER_BLOCK
            block_end = block_start + TRIALS_PER_BLOCK
            block_events = self.events[block_start:block_end]

            for t in range(1, TRIALS_PER_BLOCK):
                prev = block_events[t - 1]
                curr = block_events[t]

                # Skip if any response missing (current or previous)
                if (prev.player1_response == Response.NO_RESPONSE or curr.player1_response == Response.NO_RESPONSE
                      or prev.player2_response == Response.NO_RESPONSE
                      or curr.player2_response == Response.NO_RESPONSE):
                    continue

                # player 1
                change_p1 = curr.player1_response != prev.player1_response

                if prev.outcome == Outcome.DRAW:
                    p1_draw.append(change_p1)
                elif prev.outcome == Outcome.PLAYER1_WINS:
                    p1_win.append(change_p1)
                else:  # PLAYER2_WINS
                    p1_lose.append(change_p1)

                # player 2
                change_p2 = curr.player2_response != prev.player2_response

                if prev.outcome == Outcome.DRAW:
                    p2_draw.append(change_p2)
                elif prev.outcome == Outcome.PLAYER2_WINS:
                    p2_win.append(change_p2)
                else:  # PLAYER1_WINS
                    p2_lose.append(change_p2)

        def percent(changes: list[bool]) -> float:
            return (sum(changes) / len(changes)) * 100 if changes else float("nan")

        return (
          [percent(p1_win), percent(p2_win)],  # after win
          [percent(p1_lose), percent(p2_lose)],  # after loss
          [percent(p1_draw), percent(p2_draw)],  # after draw
        )

    def preprocess(self, bids_root: pathlib.Path, output_dir: pathlib.Path) -> None:
        """The main preprocessing pipeline, mirroring the structure of the MATLAB loop."""
        print(f"Processing {self.id}...")
        # 1. Load Data (FieldTrip's ft_read_header + ft_preprocessing)
        bids_path = mne_bids.BIDSPath(subject=self.id.replace("sub-", ""), task="RPS", root=bids_root)
        try:
            raw = mne_bids.read_raw_bids(bids_path, verbose=False)
            raw.load_data()  # Loads data into memory for processing
        except FileNotFoundError:
            print(f"  Skipping {self.id}: BIDS file not found.")
            return

        # 2. Split and Rename Channels (MATLAB's channel selection + renaming via .lay file)
        raw_p1, raw_p2 = self._prepare_players(raw)

        # 3. Clean up the massive original Raw object immediately to free RAM
        del raw
        gc.collect()

        # 4. Interpolate Bad Channels (MATLAB's ft_channelrepair)
        self._interpolate(raw_p1, self.player1, "Player 1")
        self._interpolate(raw_p2, self.player2, "Player 2")

        # 5. Downsample (MATLAB's ft_resampledata)
        # > the original code uses 256Hz --> however, the matlab function rounds the frequency to match
        #   the epoch time boundaries (-0.2 and 5.0) --> mne doesn't do that, so we manually round
        #   ourselves to the closest, standard, and divisible by the boundaries frequency.
        raw_p1.resample(250, verbose=False)
        raw_p2.resample(250, verbose=False)

        # 6. Epoch (MATLAB's ft_preprocessing with cfg.trl)
        epochs_p1 = self._create_epochs(raw_p1)
        epochs_p2 = self._create_epochs(raw_p2)

        del raw_p1, raw_p2
        gc.collect()

        # 7. Save (MATLAB's save function)
        self._save(epochs_p1, epochs_p2, output_dir)

        del epochs_p1, epochs_p2
        gc.collect()
        print(f"  Done {self.id}. Memory cleared.\n")

    def _prepare_players(self, raw):
        """Splits data, renames channels, and applies the necessary MNE montage."""
        # Channel selection logic matching the MATLAB script's use of '2-' for P1 and '1-' for P2
        p1_chans = [ch for ch in raw.ch_names if ch.startswith("2-")]
        p2_chans = [ch for ch in raw.ch_names if ch.startswith("1-")]

        p1_idx = mne.pick_channels(raw.ch_names, p1_chans)
        p2_idx = mne.pick_channels(raw.ch_names, p2_chans)

        info_p1 = mne.pick_info(raw.info, p1_idx)
        info_p2 = mne.pick_info(raw.info, p2_idx)
        
        raw_p1 = mne.io.RawArray(raw.get_data(picks=p1_idx), info_p1, verbose=False)
        raw_p2 = mne.io.RawArray(raw.get_data(picks=p2_idx), info_p2, verbose=False)

        # Helper function to rename, apply map, and set montage (The critical function)
        def fix_names(inst, prefix, label):
            # --- 0. Compute valid channels after stripping prefix ---
            stripped_chs = {ch: ch.replace(prefix, "") for ch in inst.ch_names}
            valid_chs = [ch for ch in stripped_chs.values() if ch in BIOSEMI_CODE_TO_1020_LABEL]
            
            # --- 1. Pick only the valid channels to avoid duplicates ---
            pick_idx = [inst.ch_names.index(ch_name) for ch_name, new_name in stripped_chs.items() if new_name in valid_chs]
            inst.pick(pick_idx, verbose=False)
            
            # --- 2. Rename by stripping prefix ---
            rename_map_prefix = {ch: ch.replace(prefix, "") for ch in inst.ch_names}
            inst.rename_channels(rename_map_prefix)
            
            # --- 3. Apply BioSemi → 10-20 mapping ---
            final_map = {k: v for k, v in BIOSEMI_CODE_TO_1020_LABEL.items() if k in inst.ch_names}
            inst.rename_channels(final_map)
            
            # --- 4. Set all channels to EEG type ---
            inst.set_channel_types({ch: 'eeg' for ch in inst.ch_names}, verbose=False)
            
            # --- 5. Keep only canonical BioSemi 64 channels ---
            canonical_chs = FULL_MNE_BISEOMI_MONTAGE.ch_names
            inst.pick_channels([ch for ch in canonical_chs if ch in inst.ch_names], ordered=True, verbose=False)
            
            # --- 6. Apply the 3D montage ---
            inst.set_montage(FULL_MNE_BISEOMI_MONTAGE, match_case=False, verbose=False)
            
            print(f"  {label}: Channels fixed. Total channels now: {len(inst.ch_names)}")

        fix_names(raw_p1, "2-", "Player 1")
        fix_names(raw_p2, "1-", "Player 2")
        return raw_p1, raw_p2

    def _interpolate(self, raw, player_meta, label):
        """Handles bad channel interpolation (MATLAB's ft_channelrepair)."""
        bads = [ch for ch in player_meta.preprocessing_channels_fixed if ch in raw.ch_names]
        if bads:
            raw.info['bads'] = bads
            raw.interpolate_bads(reset_bads=True, verbose=False)
            print(f"  {label}: Interpolated {bads}")

    def _create_epochs(self, raw):
        """Converts continuous data into segmented trials (MATLAB's ft_preprocessing with cfg.trl)."""
        onset_times = [e.onset for e in self.events]
        annot = mne.Annotations(onset=onset_times,
                                duration=[0] * len(onset_times),
                                description=['trial_start'] * len(onset_times))
        raw.set_annotations(annot)
        # Create event array from annotations
        events, _ = mne.events_from_annotations(raw, verbose=False)
        # Epoching (-0.2s pre-stimulus, 5.0s post-stimulus)
        return mne.Epochs(raw, events, tmin=-0.2, tmax=5.0, baseline=(-0.2, 0), preload=False, verbose=False)

    def _save(self, ep1, ep2, output_dir):
        """Saves the final Epochs objects to disk (MATLAB's save function)."""
        pair_num = self.id.replace('sub-', '')
        out_folder = output_dir / "derivatives"
        out_folder.mkdir(parents=True, exist_ok=True)

        p1_fname = out_folder / f"pair-{pair_num}_player-1_task-RPS_eeg_epo.fif"
        p2_fname = out_folder / f"pair-{pair_num}_player-2_task-RPS_eeg_epo.fif"

        ep1.save(p1_fname, overwrite=True, verbose=False)
        ep2.save(p2_fname, overwrite=True, verbose=False)


@dataclasses.dataclass
class BidsDataset:
    bids_root: pathlib.Path
    subjects: list[Subject]

    @staticmethod
    def get_from(bids_root: pathlib.Path) -> "BidsDataset":
        subjects = []
        tsv_path = bids_root / "participants.tsv"
        with open(tsv_path, newline="") as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter="\t")
            for row in reader:
                # Excluded subjects matching the MATLAB script's exclusions: 10, 23, 24
                if row["participant_id"] not in ["sub-10", "sub-23", "sub-24"]:
                    subjects.append(Subject.from_tsv_row(bids_root, row))
        return BidsDataset(bids_root, subjects)

    def preprocess(self) -> None:
        output_path = self.bids_root
        print(f"Starting preprocessing for {len(self.subjects)} subjects.")
        print(f"Outputting processed data to: {output_path / 'derivatives'}")

        for subject in self.subjects:
            if subject.id == "sub-01":
                subject.preprocess(self.bids_root, output_path)
        # self.average_results()

    def inspect_derivatives(self) -> None:
        """
        Loads the saved epochs for the first processed subject and plots the
        Evoked Response (ERP) with Global Field Power (GFP) to check data quality.
        """
        if not self.subjects:
            print("No subjects available to inspect.")
            return

        first_subject_id = self.subjects[0].id
        pair_num = first_subject_id.replace('sub-', '')
        output_path = self.bids_root / "derivatives"
        p1_fname = output_path / f"pair-{pair_num}_player-1_task-RPS_eeg_epo.fif"

        print(f"\n--- Inspecting first derivative file: {p1_fname.name} ---")

        if not p1_fname.exists():
            print(f"File not found: {p1_fname}. Have you run preprocess() yet?")
            return

        try:
            # 1. Load the epochs file
            epochs_p1 = mne.read_epochs(p1_fname, preload=False, verbose=False)

            # 2. Compute the Evoked Response
            evoked_p1 = epochs_p1.average()

            # 3. Plot the Evoked Response with Global Field Power (GFP)
            # This plot shows the time series for all channels, and the GFP curve
            # (black line) is a good indicator of overall signal quality.
            evoked_p1.plot(spatial_colors=True,
                           gfp=True,
                           zorder='std',
                           titles=f"Player 1 Evoked Response (Subject {pair_num})",
                           window_title=f"ERP and GFP for Subject {pair_num}")

            # 4. Show the topographic map at the peak of the evoked response (e.g., around 400ms)
            evoked_p1.plot_topomap(times=[0.1, 0.4],
                                   title=f"Topographies for Subject {pair_num}",
                                   show_names=True,
                                   vmax='auto',
                                   colorbar=False)

            print("Inspection plots generated. Close the MNE windows to continue.")
            mne.viz.tight_layout()  # Ensures plots fit the screen
            mne.viz.show()

        except Exception as e:
            print(f"An error occurred during inspection: {e}")
            return

    def average_results(self) -> None:
        # Placeholder: implement averaging over time bins
        raise NotImplementedError("average_results method not implemented yet.")


# --- Utility Functions ---


def get_events_for_subject(bids_root: pathlib.Path, subject_id: str) -> list[Event]:
    events = []
    fname = bids_root / subject_id / "eeg" / f"{subject_id}_task-RPS_events.tsv"
    if fname.exists():
        with open(fname, newline="") as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter="\t")
            for row in reader:
                events.append(
                  Event(float(row["onset"]), float(row["duration"]), int(row["onset_sample"]), int(row["trial_num"]),
                        Response(int(row["player1_resp"])), Response(int(row["player2_resp"])),
                        float(row["player1_rt"]), float(row["player2_rt"]), Outcome(int(row["outcome"]))))
    return events
