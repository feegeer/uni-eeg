# The Sharknado EEG project

- Team members: Felipe, Emma, Radu

## Installation

The Sharknado project requires Python >=3.12

```sh
python -m venv .venv && source .venv/bin/activate # For bash + if you still use pip. I strongly recommend switching to uv.
pip install -e . # Consider using uv, it is like 100x faster.
aws s3 sync --no-sign-request s3://openneuro.org/ds006761 ds006761 # Will download ~78GiB - use tmux and good luck :)
```

## Formatting

To format the code, run

```sh
yapf -ir src
```

## Folder Structure
- data
  - ds006761
    - sub-01
    - sub-02
    - ...
  - results
    - preprocessed_eeg
    - lda_decoding
    - eegnet_decoding
- src
  - biosemi64.mat
  - python and jupyter files
