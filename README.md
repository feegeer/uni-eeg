# The Sharknado EEG Project 🦈🌪️

A re-analysis of Moerel et al. (2025), *"Neural decoding of competitive decision-making in Rock–Paper–Scissors"*, using MNE-Python instead of the original MATLAB/FieldTrip/CoSMoMVPA toolchain.

This is a semester project for the EEG course at the University of Stuttgart. Rather than reproducing the original pipeline step-by-step, we re-implement it independently in Python and then extend it, checking whether the paper's findings hold up under a different preprocessing and decoding stack.

**Team members:** Emma Feege, Felipe Potenza, Radu-Mihai Savancea


## What the Paper Does

Moerel et al. recorded 64-channel EEG from 31 pairs of participants playing 480 rounds of Rock–Paper–Scissors. They used multivariate decoding (LDA via CoSMoMVPA) to ask whether the EEG signal contains information about:

1. The player's own response on the current trial
2. The opponent's response on the current trial
3. The player's own response on the previous trial
4. The opponent's response on the previous trial

Key finding: players' own decisions were decodable from EEG during all task phases, and only overall match *losers* showed neural encoding of previous-trial information, suggesting that relying on past outcomes may hurt performance in a game where the optimal strategy is to be random.


## What We Do

### 1. Reproduce the behavioural statistics

Before touching the EEG data, we first verify that the behavioural patterns reported in the paper (response biases, win-stay/lose-shift tendencies, Markov chain predictability) are present in the dataset. This serves as a sanity check that we're reading the data correctly and that there's something learnable.

### 2. Preprocess the EEG in MNE-Python

The original paper used FieldTrip for preprocessing (average re-referencing, epoching, bad channel interpolation, downsampling to 256 Hz, no filtering). We re-implement this in MNE-Python, which required writing custom routines in some places (e.g., channel interpolation parameters) since an exact 1:1 match between FieldTrip and MNE is not always possible. We document where and why our pipeline deviates.

### 3. Reproduce the LDA decoding pipeline

The core decoding analysis (pseudo-trial construction, balanced cross-validation folds, channel searchlight) is re-implemented in Python, staying as close as possible to the CoSMoMVPA logic. This includes a faithful reimplementation of `cosmo_classify_lda` with its default regularisation ($\lambda = 0.01 \times \text{trace}(\Sigma)/p$), `cosmo_sample_unique` for balanced sampling, and `cosmo_chunkize` for fold assignment. The goal is to verify that the decoding results replicate before changing anything.

### 4. Extend with additional classifiers and preprocessing 

Once the reproduction is in place, we go further:

- **Additional preprocessing**: we test the effect of bandpass filtering (1–70 Hz) and a 50 Hz notch filter, and re-examine their bad channel selections.
- **Linear classifiers**: we benchmark the original LDA against three alternatives (Shrinkage LDA, Linear SVM, and L2-regularised Logistic Regression) on the exact same data and folds to test whether the decoding results are robust to classifier choice.
- **Non-linear decoding with EEGNet**: we apply EEGNet, a convolutional neural network designed for EEG, as a fundamentally different decoding approach.


## Pipeline Overview

```
Raw BDF files (BioSemi 64ch, 2048 Hz)
        │
        ▼
┌─────────────────────────┐
│  Preprocessing (MNE)    │
│  • Average re-reference │
│  • Epoch: -0.2 to 5.0 s │
│  • Bad channel interp.  │
│  • Downsample to 256 Hz │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Phase splitting & binning      │
│  • Decision (0–2 s)             │
│  • Response (2–4 s)             │
│  • Feedback (4–5 s)             │
│  • Baseline correction per part │
│  • Average into 250 ms bins     │
│    → 8 + 8 + 4 = 20 time bins   │
└────────┬────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  Decoding (per participant)          │
│  • Remove block-first trials         │
│  • 10 balanced CV folds              │
│  • Pseudo-trials: avg 4, repeat 20×  │
│  • Temporal decoding (all channels)  │
│  • Channel searchlight (4 neighbours)│
│                                      │
│  Classifiers:                        │
│    ├─ LDA (CoSMoMVPA-style)          │
│    ├─ Shrinkage LDA (Ledoit-Wolf)    │
│    ├─ Linear SVM                     │
│    ├─ Logistic Regression (L2)       │
│    └─ EEGNet (separate pipeline)     │
└──────────────────────────────────────┘
```


## Project Structure

```
uni-eeg/
├── src/                           # All source code (functions + analysis scripts)
│   ├── main.py                    # Entry point for running the whole pipeline
│   ├── data_setup.py              # Checks data folder structure correctness and which pipeline steps are complete
│   ├── dataset.py                 # Organizes the original dataset and performs preprocessing
│   ├── linear_decoding.py         # Custom implementation of all four linear decoding methods
│   ├── decoding_visualization.py  # Plots linear decoding results similarly to original paper
│   ├── non_linear_decoding.py     # Implements new EEGNet as the non-linear decoding method
│   ├── eeg_visualization.py       # Plots EEGNet decoding results
│   └── markov_analysis.py         # Markov analysis implementation
├── report/                        # Final report + necessary images
│   ├── images_report/             # Figures used in the report notebooks
│   └── report.ipynb               # Main combined report
├── pyproject.toml                 # Python package config & dependencies
├── .style.yapf                    # Code formatting rules
└── README.md                      # Intro & instructions to the project
```

The `data/` directory is not tracked by git (too large). See the installation instructions below for how to obtain it.


## Installation

**Requirements:** Python ≥ 3.12

```bash
# Clone the repo
git clone https://github.com/feegeer/uni-eeg.git
cd uni-eeg
git checkout development

# Set up environment (we recommend uv, but pip works too)
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Download the dataset from OpenNeuro (~78 GiB — use tmux!)
aws s3 sync --no-sign-request s3://openneuro.org/ds006761 data/ds006761
```

### Code formatting

We use [yapf](https://github.com/google/yapf) for consistent formatting:

```bash
yapf -ir src
```

## Usage
 
After installing and downloading the data, run the full pipeline with:
 
```bash
python src/main.py
```
 
This will check the data folder structure, preprocess EEG data, run all decoding methods, perform Markov analysis, and save all results to `data/results/`. The script detects which steps have already been completed and skips them, so it is safe to re-run after interruptions.

*Note*: Given the size of the dataset, each step can take from a few seconds up to multiple hours to run. The decoding methods are specially time-consuming.
 

### Final Folder Structure with Data in Detail

```
uni-eeg/
├── src/                           # All source code (functions + analysis scripts)
│   ├── main.py                    # Entry point for running the whole pipeline
│   ├── data_setup.py              # Checks data folder structure correctness and which pipeline steps are complete
│   ├── dataset.py                 # Organizes the original dataset and performs preprocessing
│   ├── linear_decoding.py         # Custom implementation of all four linear decoding methods
│   ├── decoding_visualization.py  # Plots linear decoding results similarly to original paper
│   ├── non_linear_decoding.py     # Implements new EEGNet as the non-linear decoding method
│   ├── eeg_visualization.py       # Plots EEGNet decoding results
│   └── markov_analysis.py         # Markov analysis implementation
├── report/                        # Final report + necessary images
│   ├── images_report/             # Figures used in the report notebooks
│   └── report.ipynb               # Main combined report
├── data/                          # All data files (original dataset, reproduction, and contributions)
│   ├── ds006761/                  # Original downloaded dataset
│   ├── results/                   # Reproduction and contribution results
│   │   ├── preprocessed_eeg/      # Preprocessed EEG *.fif files
│   │   ├── linear_decoding/       # Decoding results for all four linear decoding methods
│   │   ├── non_linear_decoding/   # Decoding results for EEGNet
│   │   └── markov_analysis/       # Results from Markov analysis
│   ├── biosemi64.mat              # Channel positions for the BioSemi 64-channel cap (MATLAB format)
│   └── biosemi64.lay              # Channel layout for the BioSemi 64-channel cap (FieldTrip text format)
├── pyproject.toml                 # Python package config & dependencies
├── .style.yapf                    # Code formatting rules
└── README.md                      # Intro & instructions to the project
```

### Results Kaggle Dataset

In case there is no time to run all the steps, we have uploaded the results from `data/results/` as a public [Kaggle dataset](https://www.kaggle.com/datasets/felpspotenza/resultseegproject).


## References

**Paper:**
> Moerel, D., Grootswagers, T., Chin, J. L. L., Ciardo, F., Nijhuis, P., Quek, G. L., Smit, S., & Varlet, M. (2025). Neural decoding of competitive decision-making in Rock–Paper–Scissors. *Social Cognitive and Affective Neuroscience, 20*(1), nsaf101. https://doi.org/10.1093/scan/nsaf101

**Dataset (OpenNeuro):**
> https://doi.org/10.18112/openneuro.ds006761.v1.0.0

**Original analysis code (OSF):**
> https://doi.org/10.17605/OSF.IO/YJXKN

**Key toolboxes used:**
- [MNE-Python](https://mne.tools/) — EEG preprocessing and data handling
- [scikit-learn](https://scikit-learn.org/) — LDA, SVM, and Logistic Regression classifiers
- [EEGNet](https://github.com/vlawhern/arl-eegnet) — convolutional neural network for EEG decoding
- [CoSMoMVPA](https://www.cosmomvpa.org/) — the MATLAB toolbox used in the original paper (our Python code reimplements its core routines)