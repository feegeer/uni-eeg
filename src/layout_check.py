import scipy.io
import matplotlib.pyplot as plt
import pathlib
import mne

def plot_3D_layout(biosemi64_mat_path: pathlib.Path) -> None:
    mat_contents = scipy.io.loadmat(biosemi64_mat_path)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(mat_contents["biosemi64"][:,0], mat_contents["biosemi64"][:,1], mat_contents["biosemi64"][:,2], color='red', marker='o', label="blub")
    for i, r in enumerate(mat_contents["biosemi64"]):
        ax.text(r[0], r[1], r[2], i + 1, ha="center", va="center", fontsize=14)
    plt.show()


def read_layout(filepath: pathlib.Path) -> list[dict[str, int | float]]:
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            parts = line.split()
            if len(parts) < 6:
                continue
            rows.append({"idx": int(parts[0]), "x": float(parts[1]), "y": float(parts[2]), "w": float(parts[3]), "h": float(parts[4]), "label": " ".join(parts[5:])})
    return rows


def plot_biosemi_layout_to_10_20(biosemi_layout_path: pathlib.Path) -> None:
    rows = read_layout(biosemi_layout_path)
    rows = [r for r in rows if r["label"] not in {"COMNT", "SCALE"}]

    _, ax = plt.subplots(figsize=(6, 6))
    ax.scatter([r["x"] for r in rows], [r["y"] for r in rows], s=200, color="lightblue")

    for r in rows:
        ax.text(r["x"], r["y"], r["label"] + f" {r["idx"]}", ha="center", va="center", fontsize=8)

    ax.set_aspect("equal")
    ax.invert_yaxis()  # often visually closer to EEG layout convention
    ax.set_title("biosemi64.lay")
    ax.axis("off")
    plt.tight_layout()
    plt.show()
