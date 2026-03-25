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
  - ds006761 (following original dataset folder structure)
  - results
    - preprocessed_eeg (.fif)
    - linear_decoding
    - non_linear_decoding
  - biosemi64.mat
  - biosemi64.lay
- src
  - dataset.py (reading dataset and metadata + preprocessing + data classes for stats plot)
  - visualization.py (move all plot functions here)
  - linear_decoding.py (detached - only uses .fif files)
  - non_linear_decoding.py (detached - only uses .fif files)
  - markov_analysis.py
  - data_structure_setup.py
  - main.py (import modules and call the main functions from each step in the pipeline - comment out/back steps depending on what you want)
- report
  - images
  - main_report.ipynb