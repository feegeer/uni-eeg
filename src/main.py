import pathlib

import subject


def main():
    bids_dataset = subject.BidsDataset.get_from(pathlib.Path("ds006761"))


if __name__ == "__main__":
    main()
