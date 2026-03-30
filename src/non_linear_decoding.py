"""
EEGNet-based non-linear decoding pipeline for the RPS dataset.
 
Applies a lightweight convolutional neural network (inspired by EEGNet)
to decode Rock/Paper/Scissors responses from 64-channel EEG data.
Complements the linear decoding pipeline by testing whether a non-linear
model captures additional structure in the neural signal.
 
Requires: torch, mne, numpy, pandas, scikit-learn, matplotlib
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import mne
import matplotlib.pyplot as plt

import data_setup


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW = 3  # Sliding window width in time bins (~750 ms)


class EEGNet(nn.Module):
    """
    Lightweight convolutional neural network for EEG decoding inspired by
    the EEGNet paper (Lawhern et al., 2018).
    
    The architecture consists of:
        - Temporal convolutions: capture short-term temporal patterns
        - Spatial convolution: integrates information across channels
        - Pooling + linear layer: classification
    
    Args:
        n_channels (int): number of EEG channels
        n_classes (int): number of output classes
    """
    def __init__(self, n_channels: int, n_classes: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, (1, 5), padding=(0, 2)),
            nn.BatchNorm2d(32), 
            nn.ELU(), 
            nn.Dropout(0.3), 
            
            nn.Conv2d(32, 64, (1, 5), padding=(0, 2)),
            nn.BatchNorm2d(64), 
            nn.ELU(), 
            nn.Dropout(0.3), 
            
            nn.Conv2d(64, 64, (n_channels, 1)),
            nn.BatchNorm2d(64), 
            nn.ELU(), 
            
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------
# Data helpers
# --------------------


def epoch_to_timebinned_array(epochs: mne.Epochs) -> np.ndarray:
    """
    Convert EEG epochs into time-binned representation (250 ms bins).
    
    The trial is split into three phases, each baseline-corrected and
    averaged into 250 ms bins:
        - Decision phase (0-2 s)   → 8 bins
        - Response phase (2-4 s)   → 8 bins
        - Feedback phase (4-5 s)   → 4 bins
    
    Args:
        epochs (mne.Epochs): preprocessed EEG epochs
    
    Returns:
        np.ndarray: shape (n_trials, n_channels, 20)
    """
    data = epochs.get_data()
    times = epochs.times

    time_windows_ab = np.column_stack([np.arange(0, 2, 0.25), np.arange(0.25, 2.25, 0.25)])
    time_windows_c = np.column_stack([np.arange(0, 1, 0.25), np.arange(0.25, 1.25, 0.25)])

    def bin_part(d, t, windows):
        out = np.zeros((d.shape[0], d.shape[1], len(windows)))
        for i, (t0, t1) in enumerate(windows):
            mask = (t > t0) & (t < t1)
            out[:, :, i] = d[:, :, mask].mean(axis=2)
        return out

    # Extract and time-shift each phase
    part_a = data[:, :, (times >= -0.2) & (times <= 2)]
    part_b = data[:, :, (times >= 1.8) & (times <= 4)]
    part_c = data[:, :, (times >= 3.8) & (times <= 5)]

    times_a = times[(times >= -0.2) & (times <= 2)]
    times_b = times[(times >= 1.8) & (times <= 4)] - 2
    times_c = times[(times >= 3.8) & (times <= 5)] - 4

    # Bin each phase
    binned_a = bin_part(part_a, times_a, time_windows_ab)
    binned_b = bin_part(part_b, times_b, time_windows_ab)
    binned_c = bin_part(part_c, times_c, time_windows_c)

    return np.concatenate([binned_a, binned_b, binned_c], axis=2)


def build_behaviour(events: pd.DataFrame) -> list[np.ndarray]:
    """
    Construct decoding target matrices for both players.
    
    Each matrix has shape (n_trials, 4) with columns:
        0: this player's current response
        1: opponent's current response
        2: this player's previous response
        3: opponent's previous response
    
    Args:
        events (pd.DataFrame): events TSV loaded from the dataset
    
    Returns:
        list[np.ndarray]: [player1_targets, player2_targets]
    """
    p1 = events["player1_resp"].values
    p2 = events["player2_resp"].values

    p1_prev = np.concatenate([[np.nan], p1[:-1]])
    p2_prev = np.concatenate([[np.nan], p2[:-1]])

    return [
      np.column_stack([p1, p2, p1_prev, p2_prev]),
      np.column_stack([p2, p1, p2_prev, p1_prev]),
    ]


def make_pseudotrials(X: np.ndarray, y: np.ndarray, n_avg: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """
    Create pseudo-trials by averaging samples of the same class.
    
    For each class, trials are shuffled and grouped into non-overlapping
    blocks of `n_avg`. Each block is averaged into a single pseudo-trial.
    This improves the signal-to-noise ratio before classification.
    
    Args:
        X (np.ndarray): EEG data (n_trials, n_channels, n_timebins)
        y (np.ndarray): class labels
        n_avg (int): number of trials to average per pseudo-trial
    
    Returns:
        tuple: (pseudo_X, pseudo_y)
    """
    X_new, y_new = [], []

    for c in np.unique(y):
        idx = np.where(y == c)[0]
        np.random.shuffle(idx)

        for i in range(0, len(idx) - n_avg, n_avg):
            sel = idx[i:i + n_avg]
            X_new.append(X[sel].mean(axis=0))
            y_new.append(c)

    return np.array(X_new), np.array(y_new)


# --------------------
# Training
# --------------------


def train_model(X: np.ndarray, y: np.ndarray, epochs: int = 15) -> tuple[EEGNet, float, dict]:
    """
    Train the EEGNet on given data with a stratified train/test split.
    
    Features are z-scored before training. The model is trained with
    Adam and cross-entropy loss.
    
    Args:
        X (np.ndarray): EEG data (n_trials, n_channels, n_timebins)
        y (np.ndarray): labels (0-indexed class integers)
        epochs (int): number of training epochs
    
    Returns:
        tuple:
            model (EEGNet): trained model
            test_acc (float): final balanced accuracy on the test set
            history (dict): training and test accuracy per epoch
    """
    # Z-score normalisation
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y)

    def to_tensor(X, y):
        return (
          torch.tensor(X[:, None, :, :], dtype=torch.float32),
          torch.tensor(y),
        )

    Xtr, ytr = to_tensor(Xtr, ytr)
    Xte, yte = to_tensor(Xte, yte)

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=32, shuffle=True)

    model = EEGNet(X.shape[1], 3).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    train_accs, test_accs = [], []

    for ep in range(epochs):
        model.train()
        correct, total = 0, 0

        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()

            pred = out.argmax(1)
            correct += (pred == yb).sum().item()
            total += len(yb)

        train_accs.append(correct / total)

        model.eval()
        with torch.no_grad():
            preds = model(Xte.to(DEVICE)).argmax(1).cpu()
        test_accs.append(balanced_accuracy_score(yte, preds))

    return model, test_accs[-1], {"train_acc": train_accs, "test_acc": test_accs}


# --------------------
# Temporal decoding
# --------------------


def temporal_decoding(data: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, list[dict], EEGNet | None]:
    """
    Perform sliding-window decoding across time.
    
    A window of `WINDOW` consecutive time bins is slid across the 20 bins.
    At each position, pseudo-trials are created and an EEGNet is trained
    from scratch.
    
    Args:
        data (np.ndarray): EEG data (n_trials, n_channels, n_timebins)
        targets (np.ndarray): class labels
    
    Returns:
        tuple:
            accs (np.ndarray): balanced accuracy at each time step
            curves (list[dict]): training histories per time step
            last_model (EEGNet | None): last trained model
    """
    accs, curves = [], []
    last_model = None

    for t in range(data.shape[2] - WINDOW + 1):
        X = data[:, :, t:t + WINDOW]
        X, y = make_pseudotrials(X, targets)

        if len(np.unique(y)) < 3:
            continue

        model, acc, hist = train_model(X, y)

        accs.append(acc)
        curves.append(hist)
        last_model = model

    return np.array(accs), curves, last_model


# --------------------
# Plotting
# --------------------


def plot_training(curves: list[dict]) -> None:
    """
    Plot training and test accuracy curves across all time-step models.
    
    Args:
        curves (list[dict]): list of training histories, each containing
            'train_acc' and 'test_acc' lists
    """
    plt.figure(figsize=(10, 5))

    for c in curves:
        plt.plot(c["train_acc"], alpha=0.5)
        plt.plot(c["test_acc"], linestyle="--", alpha=0.5)

    plt.title("EEGNet Training Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    save_path = data_setup.NON_LINEAR_DIR / "training_curves.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")


def plot_paper(all_results: list[np.ndarray]) -> None:
    """
    Plot decoding results in 4-panel format matching the original paper.
    
    Panels: Own response / Opponent's response / Own previous / Opponent's previous.
    Each panel shows mean accuracy ± SEM across participants, with task
    phases indicated by coloured background bands.
    
    Args:
        all_results (list[np.ndarray]): list of 4 arrays, each containing
            per-participant accuracy time courses
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    titles = ["Own", "Other", "Own prev", "Other prev"]

    for i, ax in enumerate(axes):
        data = np.array(all_results[i]) * 100

        if len(data) == 0:
            continue

        mean = data.mean(axis=0)
        sem = data.std(axis=0) / np.sqrt(len(data))
        t = np.linspace(0, 5, len(mean))

        ax.fill_between(t, mean - sem, mean + sem, alpha=0.2)
        ax.plot(t, mean, marker="o")
        ax.axhline(33.3, linestyle="--", color="black")

        ax.axvspan(0, 2, alpha=0.1, color="orange")  # Decision phase
        ax.axvspan(2, 4, alpha=0.1, color="red")  # Response phase
        ax.axvspan(4, 5, alpha=0.1, color="purple")  # Feedback phase

        ax.set_title(titles[i])
        ax.set_ylim(30, 42)

    plt.tight_layout()

    save_path = data_setup.NON_LINEAR_DIR / "temporal_decoding.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")


# --------------------
# Main entry point
# --------------------


def run_non_linear_decoding() -> None:
    """
    Run the EEGNet non-linear decoding pipeline.
    
    If results already exist on disk, loads them and regenerates plots.
    Otherwise, trains a model for every participant x target x time step
    and saves the results.
    
    Note: training requires a GPU for reasonable runtimes.
    """
    model_path = data_setup.NON_LINEAR_DIR / "eegnet_model.pt"
    results_path = data_setup.NON_LINEAR_DIR / "results.npy"

    # Skip training if results already exist
    if model_path.exists() and results_path.exists():
        print("EEGNet already trained → loading results...")

        results = np.load(results_path, allow_pickle=True).item()

        plot_training(results["curves"])
        plot_paper([
          np.array(results["all_results"][0]),
          np.array(results["all_results"][1]),
          np.array(results["all_results"][2]),
          np.array(results["all_results"][3]),
        ])
        print("Mean accuracy:", np.mean(results["accs"]))
        return

    print("Training EEGNet...")

    pair_ids = data_setup.SUB_IDS
    target_cols = [0, 1, 2, 3]

    all_results = {0: [], 1: [], 2: [], 3: []}
    all_curves = []
    all_accs = []
    last_model = None

    for pair in pair_ids:
        print(f"Processing pair {pair}")

        events_path = data_setup.DATA_ROOT / f"sub-{pair:02d}/eeg/sub-{pair:02d}_task-RPS_events.tsv"

        for player in [1, 2]:
            epo_path = data_setup.PREPROCESSED_DIR / f"pair-{pair:02d}_player-{player}_task-RPS_eeg_epo.fif"

            if not events_path.exists() or not epo_path.exists():
                continue

            events = pd.read_csv(events_path, sep="\t")
            epochs = mne.read_epochs(epo_path, preload=True, verbose=False)

            data = epoch_to_timebinned_array(epochs)
            behav = build_behaviour(events)[player - 1]

            for t_idx, col in enumerate(target_cols):
                targets = behav[:, col]
                mask = (~np.isnan(targets)) & (targets > 0)

                X = data[mask]
                y = targets[mask].astype(int) - 1

                if len(np.unique(y)) < 3:
                    continue

                accs, curves, model = temporal_decoding(X, y)

                if len(accs) > 0:
                    all_results[t_idx].append(accs)
                    all_accs.extend(accs)
                    all_curves.extend(curves)
                    last_model = model

    print("Mean accuracy:", np.mean(all_accs))

    # Save results
    if last_model is not None:
        torch.save(last_model, model_path)

    np.save(results_path, {"accs": all_accs, "curves": all_curves, "all_results": all_results})
    print("Saved model and results.")

    plot_training(all_curves)
    plot_paper([
      np.array(all_results[0]),
      np.array(all_results[1]),
      np.array(all_results[2]),
      np.array(all_results[3]),
    ])
