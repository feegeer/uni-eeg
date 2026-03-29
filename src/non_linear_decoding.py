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

# EEGNet Model

class EEGNet(nn.Module):
    """
    Lightweight convolutional neural network for EEG decoding inspired from the EEGNet Paper 

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



def epoch_to_timebinned_array(epochs: mne.Epochs) -> np.ndarray:
    """
    Functon converts EEG epochs into time-binned representation (250 ms bins)

    The trial is split into:
    - Decision phase (0–2 s)
    - Response phase (2–4 s)
    - Feedback phase (4–5 s)

    Each phase is averaged into 250 ms bins

    Args:
        epochs (mne.Epochs): preprocessed EEG epochs

    Returns:
        Np Array: shape (n_trials, n_channels, n_timebins=20)
    """

    data = epochs.get_data()
    times = epochs.times

    TIME_WINDOWS_AB = np.column_stack([
        np.arange(0, 2, 0.25),
        np.arange(0.25, 2.25, 0.25)
    ])

    TIME_WINDOWS_C = np.column_stack([
        np.arange(0, 1, 0.25),
        np.arange(0.25, 1.25, 0.25)
    ])

    def bin_part(d, t, windows):
        out = np.zeros((d.shape[0], d.shape[1], len(windows)))
        for i, (t0, t1) in enumerate(windows):
            mask = (t > t0) & (t < t1)
            out[:, :, i] = d[:, :, mask].mean(axis=2)
        return out

    A = data[:, :, (times >= -0.2) & (times <= 2)]
    B = data[:, :, (times >= 1.8) & (times <= 4)]
    C = data[:, :, (times >= 3.8) & (times <= 5)]

    tA = times[(times >= -0.2) & (times <= 2)]
    tB = times[(times >= 1.8) & (times <= 4)] - 2
    tC = times[(times >= 3.8) & (times <= 5)] - 4

    A = bin_part(A, tA, TIME_WINDOWS_AB)
    B = bin_part(B, tB, TIME_WINDOWS_AB)
    C = bin_part(C, tC, TIME_WINDOWS_C)

    return np.concatenate([A, B, C], axis=2)



def build_behaviour(events: pd.DataFrame) -> list[np.ndarray]:
    """
    this function constructs decoding targets for both players

    Includes:
    - current response
    - opponent response
    - previous responses

    Args:
        events (pd.DataFrame): events.tsv file

    Returns:
        list[np array]: targets for both players
    """

    p1 = events["player1_resp"].values
    p2 = events["player2_resp"].values

    p1_prev = np.concatenate([[np.nan], p1[:-1]])
    p2_prev = np.concatenate([[np.nan], p2[:-1]])

    return [
        np.column_stack([p1, p2, p1_prev, p2_prev]),
        np.column_stack([p2, p1, p2_prev, p1_prev])
    ]



def make_pseudotrials(X, y, n_avg=4):
    """
    Create pseudo-trials by averaging samples of the same class.
    Improves signal-to-noise ratio.


    """

    X_new, y_new = [], []

    for c in np.unique(y):
        idx = np.where(y == c)[0]
        np.random.shuffle(idx)

        for i in range(0, len(idx) - n_avg, n_avg):
            sel = idx[i:i+n_avg]
            X_new.append(X[sel].mean(axis=0))
            y_new.append(c)

    return np.array(X_new), np.array(y_new)


# Training function

def train_model(X: np.ndarray, y: np.ndarray, epochs: int = 15):
    """
    Training loop for the EEGNet on given data

    Args:
        X (np array): EEG data (n_trials, n_channels, n_timebins)
        y (np array): labels
        epochs (int): number of training epochs

    Returns:
        tuple:
            model (nn Module)
            test_acc (float)
            history (dict)
    """

    # Normalization
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y)

    def to_tensor(X, y):
        return (
            torch.tensor(X[:, None, :, :], dtype=torch.float32),
            torch.tensor(y)
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

        train_acc = correct / total

        model.eval()
        with torch.no_grad():
            preds = model(Xte.to(DEVICE)).argmax(1).cpu()

        test_acc = balanced_accuracy_score(yte, preds)

        train_accs.append(train_acc)
        test_accs.append(test_acc)

    return model, test_accs[-1], {
        "train_acc": train_accs,
        "test_acc": test_accs
    }

WINDOW = 3  # ~750 ms

def temporal_decoding(data, targets):
    """
    Perform sliding-window decoding across time.
    """

    accs, curves = [], []
    last_model = None

    for t in range(data.shape[2] - WINDOW + 1):

        X = data[:, :, t:t+WINDOW]

        X, y = make_pseudotrials(X, targets)

        if len(np.unique(y)) < 3:
            continue

        model, acc, hist = train_model(X, y)

        accs.append(acc)
        curves.append(hist)
        last_model = model

    return np.array(accs), curves, last_model


def plot_training(curves: list[dict]) -> None:
    """
    Plot training and test accuracy curves

    Args:
        curves (np list): list of training histories
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


def plot_paper(all_results):
    """
    Plot decoding results in 4-panel format:
    Own / Other / Own prev / Other prev
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
        
        ax.axvspan(0, 2, alpha=0.1, color="orange")   # Decision phase
        ax.axvspan(2, 4, alpha=0.1, color="red")      # Response phase
        ax.axvspan(4, 5, alpha=0.1, color="purple")   # Feedback phase

        ax.set_title(titles[i])
        ax.set_ylim(30, 42)

    plt.tight_layout()

    save_path = data_setup.NON_LINEAR_DIR / "temporal_decoding.png"
    plt.savefig(save_path)
    plt.close()

    print(f"Saved plot: {save_path}")

# Maiin Call function callable from main; keep in mind that for training a gpu is required 

def run_non_linear_decoding():
    """
    Runs EEGNet non linear decoding pipeline

    - Trains model if not already trained
    - Otherwise loads results and plots

    Uses:
        data_setup paths for input/output
    """

    model_path = data_setup.NON_LINEAR_DIR / "eegnet_model.pt"
    results_path = data_setup.NON_LINEAR_DIR / "results.npy"

    # SKIP Mechanism if files already exist 
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

    TARGET_COLS = [0, 1, 2, 3]

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

            for t_idx, col in enumerate(TARGET_COLS):

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

    np.save(results_path, {
        "accs": all_accs,
        "curves": all_curves,
        "all_results": all_results
    })

    print("Saved model and results.")

    plot_training(all_curves)
    plot_paper([
        np.array(all_results[0]),
        np.array(all_results[1]),
        np.array(all_results[2]),
        np.array(all_results[3]),
    ])