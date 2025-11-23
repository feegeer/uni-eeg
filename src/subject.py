from collections.abc import Mapping
import csv
import dataclasses
import enum
import pathlib

import mne
import mne_bids
import numpy.typing as npt

from typing import Any


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
    player1_raw: mne.io.Raw
    player2_raw: mne.io.Raw

    @staticmethod
    def from_tsv_row(bids_root, row: Mapping[Any]) -> "Subject":
        player1 = Player(Gender(row["player1_gender"]), int(row["player1_age"]), Handedness(row["player1_handedness"]),
                         [ch.strip() for ch in row["player1_pre_processing_channels_fixed"].split(",") if ch.strip()])
        player2 = Player(Gender(row["player2_gender"]), int(row["player2_age"]), Handedness(row["player2_handedness"]),
                         [ch.strip() for ch in row["player2_pre_processing_channels_fixed"].split(",") if ch.strip()])
        events = get_events_for_subject(bids_root, row["participant_id"])

        bids_path = mne_bids.BIDSPath(subject=row["participant_id"].replace("sub-", ""), task="RPS", root=bids_root)

        # the raws are huge (> 3GB), we should consider loading them lazily.
        raw = mne_bids.read_raw_bids(bids_path)

        # channel names 1-... for player one, 2-... for player two
        player1_channels = [channel for channel in raw.ch_names if channel.startswith("1-")]
        player2_channels = [channel for channel in raw.ch_names if channel.startswith("2-")]

        # split up the raw data, per player
        # it might make sense to save them to disk once to make further runs faster
        player1_raw = raw.copy().pick(player1_channels)
        player2_raw = raw.copy().pick(player2_channels)

        return Subject(row["participant_id"], player1, player2, events, player1_raw, player2_raw)

    def preprocess(self) -> None:
        # Maybe throw away, since we want to plot all intermediate steps in a Jupyter notebook later on
        # interpolate noisy channels based on neighbouring channels with a distance measure of 0.5 cm
        self.interpolate_noisy_channels()
        # downsample data to 256 Hz
        self.downsample()
        # make three separate epochs for each trial (decision screen, response screen, feedback screen)
        self.epoch()
        # make baseline corrections for each epoch using a window from -200 ms to 0 ms
        self.make_baseline_corrections()

    def interpolate_noisy_channels(self) -> None:
        ...

    def downsample(self) -> None:
        ...

    def epoch(self) -> None:
        ...

    def make_baseline_corrections(self) -> None:
        ...


@dataclasses.dataclass
class BidsDataset:
    bids_root: pathlib.Path
    subjects: list[Subject]
    crop_interval: tuple[float, float] = (-200, 5000)
    interactive: bool = True

    @staticmethod
    def get_from(bids_root: pathlib.Path) -> "BidsDataset":
        subjects = []
        with open(bids_root / "participants.tsv", newline='') as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t')
            for row in reader:
                # According to https://osf.io/yjxkn/files/5dq83, subjects 10, 23 and 24 were excluded from the analysis
                if row["participant_id"] not in ["sub-10", "sub-23", "sub-24"]:
                    subjects.append(Subject.from_tsv_row(bids_root, row))
        return BidsDataset(bids_root, subjects)

    def preprocess(self) -> None:
        for subject in self.subjects:
            subject.preprocess()
        # average the resulting data into 250ms time bins (20 time bins in total for 0-5000ms time course)
        self.average_results()

    def average_results(self) -> None:
        ...


def get_events_for_subject(bids_root: pathlib.Path, subject_id: str) -> list[Event]:
    events = []
    with open(pathlib.Path(bids_root / subject_id / "eeg" / f"{subject_id}_task-RPS_events.tsv"),
              newline='') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            events.append(
              Event(float(row["onset"]), float(row["duration"]), int(row["onset_sample"]), int(row["trial_num"]),
                    Response(int(row["player1_resp"])), Response(int(row["player2_resp"])), float(row["player1_rt"]),
                    float(row["player2_rt"]), Outcome(int(row["outcome"]))))
    return events
