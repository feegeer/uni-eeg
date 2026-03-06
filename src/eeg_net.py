import os
import gc
import numpy as np
import pandas as pd
import mne
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ==========================================================
# CONFIG
# ==========================================================

N_SPLITS = 10
RANDOM_STATE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ==========================================================
# EEGNET
# ==========================================================

class EEGNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_samples):
        super().__init__()

        F1 = 8
        D = 2
        F2 = F1 * D
        kernel_length = 32

        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length),
                      padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1)
        )

        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1),
                      groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        )

        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16),
                      padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            x = self.temporal(dummy)
            x = self.depthwise(x)
            x = self.separable(x)
            self.flat_dim = x.numel()

        self.classifier = nn.Linear(self.flat_dim, n_classes)

    def forward(self, x):
        x = self.temporal(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = x.reshape(x.size(0), -1)
        return self.classifier(x)


# ==========================================================
# TRAINING
# ==========================================================

def train_fold(X_train, y_train, X_test, y_test, n_classes):

    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

    y_train = torch.tensor(y_train - 1, dtype=torch.long)
    y_test = torch.tensor(y_test - 1, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=16, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test),
                             batch_size=64)

    model = EEGNet(
        n_channels=X_train.shape[2],
        n_classes=n_classes,
        n_samples=X_train.shape[3]
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(),
                           lr=1e-3,
                           weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    patience = 15
    counter = 0

    epoch_bar = tqdm(range(100),
                     leave=False,
                     desc="      Training")

    for epoch in epoch_bar:

        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        acc = correct / total
        epoch_bar.set_postfix(val_acc=f"{acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    return best_acc


# ==========================================================
# STRICT PHASE SPLIT + 250ms WINDOWS
# ==========================================================

def split_and_bin_epochs(epochs):

    partA = epochs.copy().crop(tmin=-0.2, tmax=2.0)
    partB = epochs.copy().crop(tmin=1.8, tmax=4.0)
    partC = epochs.copy().crop(tmin=3.8, tmax=5.0)

    partA.apply_baseline((-0.2, 0.0), verbose=False)
    partB.apply_baseline((1.8, 2.0), verbose=False)
    partC.apply_baseline((3.8, 4.0), verbose=False)

    A_data, A_times = partA.get_data(), partA.times
    B_data, B_times = partB.get_data(), partB.times
    C_data, C_times = partC.get_data(), partC.times

    bin_edges = np.arange(0, 5.0001, 0.25)
    all_bins = []

    for i in range(len(bin_edges) - 1):

        t0, t1 = bin_edges[i], bin_edges[i + 1]

        if t0 < 2:
            mask = (A_times > t0) & (A_times < t1)
            all_bins.append(A_data[:, :, mask])
        elif t0 < 4:
            mask = (B_times > t0) & (B_times < t1)
            all_bins.append(B_data[:, :, mask])
        else:
            mask = (C_times > t0) & (C_times < t1)
            all_bins.append(C_data[:, :, mask])

    all_bins = np.array(all_bins)
    return np.transpose(all_bins, (1, 2, 3, 0))


# ==========================================================
# TEMPORAL DECODING
# ==========================================================

def temporal_decoding_eegnet(X, y):

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    n_bins = X.shape[3]
    acc = np.zeros(n_bins)
    n_classes = len(np.unique(y))

    fold_bar = tqdm(skf.split(X, y),
                    total=N_SPLITS,
                    desc="    CV folds",
                    leave=False)

    for fold_idx, (train_idx, test_idx) in enumerate(fold_bar):

        X_train_raw, y_train = X[train_idx], y[train_idx]
        X_test_raw, y_test = X[test_idx], y[test_idx]

        for t in range(n_bins):

            X_train = X_train_raw[:, :, :, t]
            X_test = X_test_raw[:, :, :, t]

            fold_acc = train_fold(
                X_train,
                y_train,
                X_test,
                y_test,
                n_classes
            )

            acc[t] += fold_acc

    return acc / N_SPLITS


# ==========================================================
# MAIN PIPELINE
# ==========================================================

def run_pipeline(path_to_data, results_dir):

    os.makedirs(results_dir, exist_ok=True)

    pairs_removed = [10, 23, 24]
    pair_ids = [i for i in range(1, 35) if i not in pairs_removed]

    for pair in tqdm(pair_ids, desc="Pairs"):

        tsv_path = os.path.join(
            path_to_data,
            f"sub-{pair:02d}",
            "eeg",
            f"sub-{pair:02d}_task-RPS_events.tsv"
        )

        df = pd.read_csv(tsv_path, sep="\t")
        events = df.iloc[:, [4, 6, 8]].values

        p1 = np.zeros((len(events), 5))
        p1[:, :3] = events
        p1[1:, 3:] = events[:-1, :2]

        p2 = np.zeros((len(events), 5))
        p2[:, 0], p2[:, 1] = events[:, 1], events[:, 0]
        p2[:, 2] = events[:, 2]
        p2[events[:, 2] == 2, 2] = 3
        p2[events[:, 2] == 3, 2] = 2
        p2[1:, 3:] = p2[:-1, :2]

        for ppt, behav in enumerate([p1, p2], start=1):

            print(f"\n  Player {ppt}")

            epo_path = os.path.join(
                path_to_data,
                "new_derivatives",
                f"pair-{pair:02d}_player-{ppt}_task-RPS_eeg_epo.fif"
            )

            if not os.path.exists(epo_path):
                continue

            epochs = mne.read_epochs(epo_path, preload=True, verbose=False)
            epochs.set_eeg_reference("average", verbose=False)

            rem_idx = np.arange(0, 480, 40)
            epochs.drop(rem_idx, verbose=False)
            behav = np.delete(behav, rem_idx, axis=0)

            X = split_and_bin_epochs(epochs)

            results_file = os.path.join(
                results_dir,
                f"pair-{pair:02d}_player-{ppt}_eegnet_decoding.h5"
            )

            with h5py.File(results_file, "w") as hf:

                targets = {"self": 0, "other": 1,
                           "selfp": 3, "otherp": 4}

                for name, col in tqdm(targets.items(),
                                      desc="  Targets",
                                      leave=False):

                    y = behav[:, col]
                    valid = (~np.isnan(y)) & (y > 0)

                    if not np.any(valid):
                        continue

                    Xv, yv = X[valid], y[valid]

                    print(f"    EEGNet decoding {name}")

                    temp_acc = temporal_decoding_eegnet(Xv, yv)

                    grp = hf.create_group(name)
                    grp.create_dataset("temporal_accuracy",
                                       data=temp_acc)

            gc.collect()


if __name__ == "__main__":

    PATH_TO_DATA = "src/ds006761"
    RESULTS_DIR = os.path.join(
        PATH_TO_DATA,
        "new_derivatives",
        "eegnet_replication"
    )

    run_pipeline(PATH_TO_DATA, RESULTS_DIR)