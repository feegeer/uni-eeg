import pathlib

import subject


def main():
    # bids_dataset = subject.BidsDataset.get_from(
    #     pathlib.Path(r"C:\Users\poten\uni-eeg\src\ds006761")
    # )
    bids_dataset = subject.BidsDataset.get_from(pathlib.Path("ds006761"))
    bids_dataset.preprocess()


if __name__ == "__main__":
    main()
