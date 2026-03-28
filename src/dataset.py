import collections
from collections.abc import Mapping
import csv
import dataclasses
import enum
from functools import lru_cache
import gc
import math
import pathlib
from typing import Any

import mne
import mne_bids
import numpy as np
import scipy.io

# --- Global Constants ---
# The standard BioSemi codes in order (A1-A32 then B1-B32) corresponding
# to the channel sequence used by FieldTrip's biosemi64.lay template.
BIOSEMI_ORDERED_CODES = [
  'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18',
  'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30', 'A31', 'A32', 'B1', 'B2', 'B3',
  'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20',
  'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B28', 'B29', 'B30', 'B31', 'B32'
]


def _biosemi64_mat_path() -> pathlib.Path:
    """Absolute path to the bundled BioSemi 3D coordinates matrix."""
    return pathlib.Path("data/biosemi64.mat")


@lru_cache(maxsize=1)
def ten_twenty_labels() -> list[str]:
    """Return the BioSemi64 channel names in MNE's built-in 10-20 montage order."""
    montage = mne.channels.make_standard_montage("biosemi64")
    return montage.ch_names[:64]


@lru_cache(maxsize=1)
def biosemi_coords_3d() -> np.ndarray:
    """Load BioSemi64 3D coordinates from the local `.mat` file."""
    mat_path = _biosemi64_mat_path()
    if not mat_path.exists():
        raise FileNotFoundError(f"biosemi64.mat not found at {mat_path}. ")
    mat_contents = scipy.io.loadmat(str(mat_path))
    return mat_contents["biosemi64"]


@lru_cache(maxsize=1)
def full_mne_biosemi_montage() -> mne.channels.DigMontage:
    """Create an MNE DigMontage for the BioSemi64 3D coordinates."""
    labels = ten_twenty_labels()
    coords = biosemi_coords_3d()
    ch_pos = dict(zip(labels, coords))
    return mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")


@lru_cache(maxsize=1)
def biosemi_distance_matrix() -> np.ndarray:
    """Pairwise Euclidean distances between BioSemi64 channels."""
    coords = biosemi_coords_3d()
    return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)


@lru_cache(maxsize=1)
def fieldtrip_neighbors() -> dict[str, list[str]]:
    """Neighbour definition approximating FieldTrip's BioSemi64 template."""
    labels = ten_twenty_labels()
    dist = biosemi_distance_matrix()
    neighbors: dict[str, list[str]] = {}
    for i, ch in enumerate(labels):
        neighbor_index = [j for j in range(len(labels)) if j != i and dist[i, j] < 0.5]
        neighbors[ch] = [labels[j] for j in neighbor_index]
    return neighbors


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
        raw = self.read_raw_eeg_data(bids_root)

        # 2. Split and Rename Channels (MATLAB's channel selection + renaming via .lay file)
        raw_p1, raw_p2 = self.prepare_players(raw)

        # 3. Clean up the massive original Raw object immediately to free RAM
        del raw
        gc.collect()

        # raw_p1.filter(l_freq=1.0, h_freq=35.0)
        # raw_p2.filter(l_freq=1.0, h_freq=35.0)

        # 6. Epoch (MATLAB's ft_preprocessing with cfg.trl)
        epochs_p1 = self.epoch_players(raw_p1)
        epochs_p2 = self.epoch_players(raw_p2)

        del raw_p1, raw_p2
        gc.collect()

        # 4. Interpolate Bad Channels (MATLAB's ft_channelrepair equivalent)
        self.interpolate(epochs_p1, self.player1, "Player 1")
        self.interpolate(epochs_p2, self.player2, "Player 2")

        if not epochs_p1.preload:
            epochs_p1.load_data()
        epochs_p1.resample(256, verbose=False)

        if not epochs_p2.preload:
            epochs_p2.load_data()
        epochs_p2.resample(256, verbose=False)

        # 7. Save (MATLAB's save function)
        self.save(epochs_p1, epochs_p2, output_dir)

        del epochs_p1, epochs_p2
        gc.collect()
        print(f"  Done {self.id}. Memory cleared.\n")

    def read_raw_eeg_data(self, bids_root: pathlib.Path) -> mne.io.Raw:
        bids_path = mne_bids.BIDSPath(subject=self.id.replace("sub-", ""), task="RPS", root=bids_root)
        try:
            raw = mne_bids.read_raw_bids(bids_path, verbose=False)
            raw.load_data()  # Loads data into memory for processing
        except FileNotFoundError:
            print(f"  Skipping {self.id}: BIDS file not found.")
            return
        return raw

    def prepare_players(self, raw: mne.io.Raw) -> tuple[mne.io.Raw, mne.io.Raw]:
        player_raws = []

        labels_1020 = ten_twenty_labels()
        montage = full_mne_biosemi_montage()

        for player_prefix in ["2-", "1-"]:
            expected_chs = [f"{player_prefix}{code}" for code in BIOSEMI_ORDERED_CODES]

            missing = [ch for ch in expected_chs if ch not in raw.ch_names]
            if missing:
                raise ValueError(f"Missing expected EEG channels for {player_prefix}: {missing}")

            player_raw = raw.copy().pick(expected_chs)
            rename_map = dict(zip(expected_chs, labels_1020))
            player_raw.rename_channels(rename_map)

            player_raw.set_channel_types({ch: "eeg" for ch in player_raw.ch_names}, verbose=False)
            player_raw.set_montage(montage, match_case=False, verbose=False)
            player_raws.append(player_raw)
        return player_raws[0], player_raws[1]

    def epoch_players(self, raw, sample_frequency: int = 2048):
        onset_samples = np.array([e.onset_sample for e in self.events], dtype=int)

        prestim_samp = math.ceil(0.2 * sample_frequency)  # 410
        poststim_samp = math.ceil(5.0 * sample_frequency)  # 10240

        # each row is [sample ID, dummy value for previous value, event ID (all events are of the same type, so dummy value 1 is ok here)]
        events = np.column_stack(
          [onset_samples,
           np.zeros(len(onset_samples), dtype=int),
           np.ones(len(onset_samples), dtype=int)])

        tmin = -prestim_samp / sample_frequency
        tmax = poststim_samp / sample_frequency

        epochs = mne.Epochs(raw, events, event_id={"trial_start": 1}, tmin=tmin, tmax=tmax, baseline=None)
        return epochs

    def interpolate(self, epochs, player_meta, label):
        """
        Closer match to FieldTrip ft_channelrepair(method='weighted'):

        1. Build a full channel x channel repair matrix.
        2. Replace each bad-channel row with inverse-distance weights on good neighbours.
        3. Apply that same repair matrix to each epoch in one shot.

        Operates in place on an MNE Epochs object.
        """
        bads = [ch for ch in player_meta.preprocessing_channels_fixed if ch in epochs.ch_names]
        if not bads:
            return

        if not epochs.preload:
            epochs.load_data()

        data = epochs.get_data(copy=False)  # shape: (n_epochs, n_channels, n_times)
        ch_names = epochs.ch_names
        ch_to_idx = {ch: i for i, ch in enumerate(ch_names)}
        bad_set = set(bads)

        n_channels = len(ch_names)
        repair = np.eye(n_channels, dtype=float)

        neighbors = fieldtrip_neighbors()
        dist_matrix = biosemi_distance_matrix()

        unable = []

        for bad_ch in bads:
            bad_idx = ch_to_idx[bad_ch]

            # FieldTrip starts from the neighbour definition and removes bad channels
            good_neighs = [ch for ch in neighbors[bad_ch] if ch in ch_to_idx and ch not in bad_set]
            print(f"good neighbours of {bad_ch}: {good_neighs}")

            # Zero out self-copying for this bad channel
            repair[bad_idx, bad_idx] = 0.0

            if len(good_neighs) == 0:
                unable.append(bad_idx)
                continue

            neigh_idx = np.array([ch_to_idx[ch] for ch in good_neighs], dtype=int)

            # Distances from this bad channel to its good neighbours
            distances_to_good_neighbors = dist_matrix[bad_idx, neigh_idx]

            # Inverse-distance weights
            if np.any(distances_to_good_neighbors == 0):
                weights = np.zeros_like(distances_to_good_neighbors, dtype=float)
                weights[distances_to_good_neighbors == 0] = 1.0 / np.sum(distances_to_good_neighbors == 0)
            else:
                weights = 1.0 / distances_to_good_neighbors
                weights = weights / weights.sum()

            # Replace the bad-channel row with neighbour weights
            repair[bad_idx, :] = 0.0
            repair[bad_idx, neigh_idx] = weights

        # Apply the same repair matrix to every epoch:
        # For each epoch, (channels x channels) @ (channels x time) -> (channels x time)
        repaired_data = np.einsum("ij,ejt->eit", repair, data)

        # FieldTrip sets unreconstructable bad channels to NaN
        if unable:
            repaired_data[:, unable, :] = np.nan

        data[:] = repaired_data

        epochs.info["bads"] = []
        print(f"  {label}: Interpolated {bads} with FieldTrip-style weighted neighbours")

    def save(self, ep1, ep2, output_dir):
        """Saves the final Epochs objects to disk (MATLAB's save function)."""
        pair_num = self.id.replace('sub-', '')
        output_dir.mkdir(parents=True, exist_ok=True)

        p1_fname = output_dir / f"pair-{pair_num}_player-1_task-RPS_eeg_epo.fif"
        p2_fname = output_dir / f"pair-{pair_num}_player-2_task-RPS_eeg_epo.fif"

        ep1.save(p1_fname, overwrite=True, verbose=False)
        ep2.save(p2_fname, overwrite=True, verbose=False)


@dataclasses.dataclass
class BidsDataset:
    bids_root: pathlib.Path
    subjects: list[Subject]
    output_path: pathlib.Path

    @staticmethod
    def get_from(bids_root: pathlib.Path,
                 output_path: pathlib.Path,
                 exclude_subjects: list[str] = ["sub-10", "sub-23", "sub-24"]) -> "BidsDataset":
        subjects = []
        tsv_path = bids_root / "participants.tsv"
        with open(tsv_path, newline="") as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter="\t")
            for row in reader:
                # Excluded subjects matching the MATLAB script's exclusions: 10, 23, 24
                if row["participant_id"] not in exclude_subjects:
                    subjects.append(Subject.from_tsv_row(bids_root, row))
        return BidsDataset(bids_root, subjects, output_path)

    def preprocess(self) -> None:
        print(f"Starting preprocessing for {len(self.subjects)} subjects.")
        print(f"Outputting processed data to: {self.output_path}")

        sub_completed = []
        for i in range(21, 35):
            sub_completed.append(f"sub-{i:02d}")
        for subject in self.subjects:
            if subject.id in sub_completed:
                subject.preprocess(self.bids_root, self.output_path)
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
        p1_fname = self.output_path / f"pair-{pair_num}_player-1_task-RPS_eeg_epo.fif"

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
