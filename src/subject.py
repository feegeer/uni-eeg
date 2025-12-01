import collections
from collections.abc import Mapping
import csv
import dataclasses
import enum
import pathlib
import gc  # for garbage collection
import numpy as np
import mne
import mne_bids
import numpy.typing as npt

from typing import Any


# --- Constants ---
# Dictionary to map proprietary BioSemi channel codes (e.g., 'A1') 
# to the standardized 10-20 EEG system names (e.g., 'Fp1').
# COMPARISON (MATLAB): This is needed BEFORE montage application in MNE. In the MATLAB script, 
# the channel labels were updated (data_epoch.label = layout.label) *after* loading and epoching, 
# relying on the BioSemi codes for the initial data selection.
BIOSEMI_TO_1020 = {
    'A1': 'Fp1', 'A2': 'AF7', 'A3': 'AF3', 'A4': 'F1', 'A5': 'F3', 'A6': 'F5', 'A7': 'F7', 'A8': 'FT7',
    'A9': 'FC5', 'A10': 'FC3', 'A11': 'FC1', 'A12': 'C1', 'A13': 'C3', 'A14': 'C5', 'A15': 'T7', 'A16': 'TP7',
    'A17': 'CP5', 'A18': 'CP3', 'A19': 'CP1', 'A20': 'P1', 'A21': 'P3', 'A22': 'P5', 'A23': 'P7', 'A24': 'P9',
    'A25': 'PO7', 'A26': 'PO3', 'A27': 'O1', 'A28': 'Iz', 'A29': 'Oz', 'A30': 'POz', 'A31': 'Pz', 'A32': 'CPz',
    'B1': 'Fpz', 'B2': 'Fp2', 'B3': 'AF8', 'B4': 'AF4', 'B5': 'AFz', 'B6': 'Fz', 'B7': 'F2', 'B8': 'F4',
    'B9': 'F6', 'B10': 'F8', 'B11': 'FT8', 'B12': 'FC6', 'B13': 'FC4', 'B14': 'FC2', 'B15': 'FCz', 'B16': 'Cz',
    'B17': 'C2', 'B18': 'C4', 'B19': 'C6', 'B20': 'T8', 'B21': 'TP8', 'B22': 'CP6', 'B23': 'CP4', 'B24': 'CP2',
    'B25': 'P2', 'B26': 'P4', 'B27': 'P6', 'B28': 'P8', 'B29': 'P10', 'B30': 'PO8', 'B31': 'PO4', 'B32': 'O2'
}

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
        p1_bad = [ch.strip() for ch in row["player1_pre_processing_channels_fixed"].split(",") if ch.strip()]
        player1 = Player(Gender(row["player1_gender"]), int(row["player1_age"]), Handedness(row["player1_handedness"]), p1_bad)

        p2_bad = [ch.strip() for ch in row["player2_pre_processing_channels_fixed"].split(",") if ch.strip()]
        player2 = Player(Gender(row["player2_gender"]), int(row["player2_age"]), Handedness(row["player2_handedness"]), p2_bad)

        events = get_events_for_subject(bids_root, pid)

        return Subject(pid, player1, player2, events)


    def preprocess(self, bids_root: pathlib.Path, output_dir: pathlib.Path) -> None:
        """
        Loads Raw Data (Lazy Load) -> Fixes Channels -> Epochs -> Saves -> Clears Memory.
        """
        print(f"Processing {self.id}...")
        # 1. Load Data
        bids_path = mne_bids.BIDSPath(subject=self.id.replace("sub-", ""), task="RPS", root=bids_root)
        print("birds_path:", bids_path)
        try:
            raw = mne_bids.read_raw_bids(bids_path, verbose=False)
            raw.load_data() # Loads data into memory for processing
        except FileNotFoundError:
            print(f"  Skipping {self.id}: BIDS file not found.")
            return
        # 2. Split and Rename Channels
        raw_p1, raw_p2 = self._prepare_players(raw)
        # 3. Clean up the massive original Raw object immediately
        del raw
        # 4. Interpolate Bad Channels
        self._interpolate(raw_p1, self.player1, "Player 1")
        self._interpolate(raw_p2, self.player2, "Player 2")
        # 5. Downsample (In place to save RAM)
        raw_p1.resample(256, verbose=False)
        raw_p2.resample(256, verbose=False)
        # 6. Epoch
        epochs_p1 = self._create_epochs(raw_p1)
        epochs_p2 = self._create_epochs(raw_p2)
        # 7. Save
        self._save(epochs_p1, epochs_p2, output_dir)
        # 8. Force Garbage Collection
        del raw_p1, raw_p2, epochs_p1, epochs_p2
        gc.collect() 
        print(f"  Done {self.id}. Memory cleared.\n")

    def _prepare_players(self, raw):
        # Separate channels based on the '1-' and '2-' prefixes
        p1_chans = [ch for ch in raw.ch_names if ch.startswith("2-")]
        p2_chans = [ch for ch in raw.ch_names if ch.startswith("1-")]
        
        # Get channel indices for efficient data access
        p1_idx = mne.pick_channels(raw.ch_names, p1_chans)
        p2_idx = mne.pick_channels(raw.ch_names, p2_chans)

        # Create new info and RawArray objects directly from data slices (MEMORY FIX)
        info_p1 = mne.pick_info(raw.info, p1_idx)
        info_p2 = mne.pick_info(raw.info, p2_idx)
        
        # This extracts only the necessary data array, avoiding creation of intermediate 8GB copies.
        raw_p1 = mne.io.RawArray(raw.get_data(picks=p1_idx), info_p1, verbose=False)
        raw_p2 = mne.io.RawArray(raw.get_data(picks=p2_idx), info_p2, verbose=False)

        # Helper function to rename, apply map, and set montage (The critical function)
        def fix_names(inst, prefix):
            # 1. Strip player prefix ('2-A1' -> 'A1')
            rename_map_prefix = {ch: ch.replace(prefix, "") for ch in inst.ch_names}
            inst.rename_channels(rename_map_prefix)
            
            # 2. Map BioSemi code to 10-20 name ('A15' -> 'T7')
            final_map = {k: v for k, v in BIOSEMI_TO_1020.items() if k in inst.ch_names}
            if final_map:
                inst.rename_channels(final_map) 
            
            # 3. CRITICAL: Set channel types to 'eeg' and pick ONLY the EEG channels
            inst.set_channel_types({ch: 'eeg' for ch in inst.ch_names}, verbose=False) 
            
            # Use the canonical list of channels from the 10-20 system (64 channels)
            montage_1020 = mne.channels.make_standard_montage("standard_1020")
            
            # Drop any non-EEG/unmapped channels (e.g., EOGs, references)
            eeg_chans = [ch for ch in inst.ch_names if ch in montage_1020.ch_names]
            inst.pick_channels(eeg_chans, ordered=True, verbose=False)
            
            # 4. Apply Montage (The Most Robust Way)
            # Create a dictionary of positions only for the channels that survived the picking.
            ch_pos_dict = {
                ch_name: pos 
                for ch_name, pos in montage_1020.get_positions()['ch_pos'].items() 
                if ch_name in inst.ch_names
            }
            
            # Use mne.channels.make_dig_montage to create the montage object.
            # This is more robust than set_montage when starting from RawArray.
            dig_montage = mne.channels.make_dig_montage(ch_pos=ch_pos_dict, coord_frame='head')
            
            # Apply the manually created Digital Montage
            inst.set_montage(dig_montage, match_case=False, verbose=False)

        fix_names(raw_p1, "2-")
        fix_names(raw_p2, "1-")
        return raw_p1, raw_p2

    def _interpolate(self, raw, player_meta, label):
        # Only interpolate channels that exist in the raw data
        bads = [ch for ch in player_meta.preprocessing_channels_fixed if ch in raw.ch_names]
        if bads:
            raw.info['bads'] = bads
            raw.interpolate_bads(reset_bads=True, verbose=False)
            print(f"  {label}: Interpolated {bads}")

    def _create_epochs(self, raw):
        # The events.tsv contains onset times, but MNE needs an annotation object
        onset_times = [e.onset for e in self.events]
        annot = mne.Annotations(onset=onset_times, duration=[0]*len(onset_times), description=['trial_start']*len(onset_times))
        raw.set_annotations(annot)
        
        # Create event array from annotations
        events, _ = mne.events_from_annotations(raw, verbose=False)
        
        # Epoching (-0.2s pre-stimulus, 5.0s post-stimulus)
        return mne.Epochs(raw, events, tmin=-0.2, tmax=5.0, baseline=(-0.2, 0), preload=True, verbose=False)

    def _save(self, ep1, ep2, output_dir):
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
    # crop_interval: tuple[float, float] = (-200, 5000)
    # interactive: bool = True

    @staticmethod
    def get_from(bids_root: pathlib.Path) -> "BidsDataset":
        subjects = []
        tsv_path = bids_root / "participants.tsv"
        with open(tsv_path, newline="") as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter="\t")
            for row in reader:
                # According to https://osf.io/yjxkn/files/5dq83, subjects 10, 23 and 24 were excluded from the analysis
                if row["participant_id"] not in ["sub-10", "sub-23", "sub-24"]:
                    subjects.append(Subject.from_tsv_row(bids_root, row))
        return BidsDataset(bids_root, subjects)

    def preprocess(self) -> None:
        output_path = self.bids_root
        print(f"Starting preprocessing for {len(self.subjects)} subjects.")
        print(f"Outputting processed data to: {output_path / 'derivatives'}")
        
        for subject in self.subjects:
            subject.preprocess(self.bids_root, output_path)
        # self.average_results()

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
                    Event(float(row["onset"]), float(row["duration"]), int(row["onset_sample"]), 
                          int(row["trial_num"]), Response(int(row["player1_resp"])), Response(int(row["player2_resp"])), 
                          float(row["player1_rt"]), float(row["player2_rt"]), Outcome(int(row["outcome"]))
                    )
                )
    return events
