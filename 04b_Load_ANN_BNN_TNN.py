'''
In The Name of God

Ali Pilehvar Meibody

Last Update : 08 sep 2026


04b_Load_ANN_BNN_TNN.py


Load the three trained MNIST networks and plot comparisons only:

    ANN  — full-precision floats
    BNN  — forward weights {-1, +1}  (sign of latent weights in 07_sep_2026_BNN.pth)
    TNN  — ternary-only {-1, 0, +1}

'''



# ============================================
'''                   Imports              '''
# ============================================
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path

from paths import DATA_DIR, PROJECT_DIR, choose_checkpoint





# ============================================
'''                Variables              '''
# ============================================
SEED = 42
batch_size = 16
BATCH_SIZE = batch_size
SAVE_FIGS = False
FIGURE_DPI = 150
DEVICE = torch.device("cpu")

FIGURES_DIR = PROJECT_DIR / "figures" / "load_ann_bnn_tnn"

ACADEMIC_COLORS = {
    "ANN": "#1F4E79",  # navy
    "BNN": "#8EC4E6",  # light blue
    "TNN": "#7B2D3B",  # burgundy
}
NEG_COLOR = "#1F4E79"
ZERO_COLOR = "#8A8A8A"
POS_COLOR = "#7B2D3B"

MODEL_COLORS = ACADEMIC_COLORS

# BNN checkpoint stores only the final loss; curve taken from the 07 sep 2026 run log.
BNN_TRAIN_LOSS = [
    123.6093, 14.4832, 2.2088, 2.2430, 2.3112, 2.3025, 2.3104, 2.3080, 2.2952, 2.2934,
    2.2667, 2.2806, 2.2753, 2.2316, 2.2031, 2.2341, 2.1799, 2.2032, 2.1452, 2.2074,
    2.1752, 2.1204, 2.1175, 2.1355, 2.1435, 2.1394, 2.1067, 2.1091, 2.0654, 2.1055,
    2.1689, 2.0957, 2.0080, 2.1560, 2.0774, 2.1089, 2.0778, 2.0856, 2.0614, 2.0285,
    2.0287, 2.0951, 2.0453, 2.0417, 2.0754, 2.0779, 2.0332, 1.9997, 2.0183, 1.9983,
    2.0793, 1.9493, 1.9592, 2.0085, 1.9749, 1.9205, 2.0008, 1.9848, 2.1332, 1.9253,
    1.9505, 1.9608, 1.9421, 1.9055, 1.9259, 1.9897, 1.9520, 1.9082, 1.8763, 1.8487,
    1.8989, 1.9000, 1.8708, 1.9525, 1.8961, 1.8729, 1.8515, 1.8209, 1.8545, 1.7986,
    1.8435, 1.8094, 1.7658, 1.8275, 1.7805, 1.8175, 1.7628, 1.7571, 1.8131, 1.8038,
    1.7853, 1.7986, 1.8900, 1.7790, 1.7723, 1.7348, 1.7513, 1.8115, 1.8602, 1.7714,
]


# ============================================
# Model (same 784-256-128-10 as 01 / 02 / 03 / 04)
# ============================================
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


# ============================================
# Load helpers
# ============================================
def _load_torch_file(path: Path, device=DEVICE):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _extract_state_dict(ckpt) -> dict:
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict) and all(str(k).startswith("fc") for k in ckpt):
        return ckpt
    raise ValueError(
        "Checkpoint format not recognized. Expected 'state_dict', "
        "'model_state_dict', or raw fc*.weight keys."
    )


def _require_checkpoint(path: Path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return path


def _binarize_weights_(model: ANN) -> None:
    """Replace latent floats with sign(w) in {-1, +1} (zeros -> +1)."""
    with torch.no_grad():
        for name in ("fc1", "fc2", "fc3"):
            w = getattr(model, name).weight
            w.copy_(torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w)))


def _verify_ternary_weights(model: ANN, label: str) -> None:
    allowed = {-1.0, 0.0, 1.0}
    for name in ("fc1", "fc2", "fc3"):
        w = getattr(model, name).weight.detach()
        uniq = set(torch.unique(w).tolist())
        if not uniq.issubset(allowed):
            raise ValueError(f"{label}: {name}.weight has non-ternary values: {sorted(uniq)}")


def _verify_binary_weights(model: ANN, label: str) -> None:
    allowed = {-1.0, 1.0}
    for name in ("fc1", "fc2", "fc3"):
        w = getattr(model, name).weight.detach()
        uniq = set(torch.unique(w).tolist())
        if not uniq.issubset(allowed):
            raise ValueError(f"{label}: {name}.weight has non-binary values: {sorted(uniq)}")


def _load_from_checkpoint(path: Path, device) -> tuple:
    path = _require_checkpoint(path)
    ckpt = _load_torch_file(path, device)
    state = _extract_state_dict(ckpt)
    model = ANN().to(device)
    model.load_state_dict(state)
    model.eval()
    meta = ckpt if isinstance(ckpt, dict) else {}
    return model, meta


def load_ann(checkpoint, device=DEVICE):
    path = _require_checkpoint(checkpoint)
    model, meta = _load_from_checkpoint(path, device)
    return model, path, meta


def load_bnn(checkpoint, device=DEVICE):
    """Load BNN latent weights, then binarize to {-1, +1} for inference / plots."""
    path = _require_checkpoint(checkpoint)
    model, meta = _load_from_checkpoint(path, device)
    _binarize_weights_(model)
    _verify_binary_weights(model, path.name)
    return model, path, meta


def load_tnn_ternary(checkpoint, device=DEVICE):
    path = _require_checkpoint(checkpoint)
    model, meta = _load_from_checkpoint(path, device)
    _verify_ternary_weights(model, path.name)
    return model, path, meta


def collect_layer_info(model: ANN) -> list:
    layer_info = []
    for name in ("fc1", "fc2", "fc3"):
        layer = getattr(model, name)
        w = layer.weight.detach().cpu().numpy()
        b = layer.bias.detach().cpu().numpy()
        layer_info.append({
            "name": name,
            "weight": w,
            "bias": b,
            "weight_shape": w.shape,
        })
    return layer_info


def _print_weight_stats(layer_info: list, model_label: str, kind: str) -> None:
    print(f"\n{model_label} — weight statistics")
    print("-" * 50)
    for li in layer_info:
        w = li["weight"]
        print(f"  {li['name']} shape={li['weight_shape']}")
        if kind == "ternary":
            for v in (-1.0, 0.0, 1.0):
                n = (w == v).sum()
                print(f"    {v:+.0f}: {n:,} ({100 * n / w.size:.2f}%)")
        elif kind == "binary":
            for v in (-1.0, 1.0):
                n = (w == v).sum()
                print(f"    {v:+.0f}: {n:,} ({100 * n / w.size:.2f}%)")
        else:
            uniq = np.unique(w)
            print(f"    unique values: {uniq.size:,}")
            print(f"    mean={w.mean():+.6f}  std={w.std():.6f}  "
                  f"min={w.min():+.6f}  max={w.max():+.6f}")


def print_model_summary(model: ANN, name: str, ckpt_path: Path, meta: dict, acc=None) -> None:
    n_params = sum(p.numel() for p in model.parameters())
    saved = meta.get("test_accuracy")
    saved_str = f"{saved:.2f}%" if saved is not None else "n/a"
    live_str = f"{acc:.2f}%" if acc is not None else "n/a"
    print(f"\n{name}")
    print(f"  checkpoint: {ckpt_path.name}")
    print(f"  parameters: {n_params:,}")
    print(f"  saved test accuracy: {saved_str}")
    print(f"  evaluated test accuracy: {live_str}")
    print(f"  layers: fc1 {tuple(model.fc1.weight.shape)}, "
          f"fc2 {tuple(model.fc2.weight.shape)}, fc3 {tuple(model.fc3.weight.shape)}")


# ============================================
# Data / evaluation
# ============================================
def make_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    test_dataset = datasets.MNIST(
        root=str(DATA_DIR), train=False, transform=transform, download=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return test_loader


def evaluate_accuracy(model, test_loader):
    model.eval()
    correct = total = 0
    class_correct = np.zeros(10, dtype=np.int64)
    class_total = np.zeros(10, dtype=np.int64)
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for c in range(10):
                mask = labels == c
                class_total[c] += int(mask.sum().item())
                if mask.any():
                    class_correct[c] += int((preds[mask] == labels[mask]).sum().item())
    per_class = {
        c: (100.0 * class_correct[c] / class_total[c] if class_total[c] else 0.0)
        for c in range(10)
    }
    return 100.0 * correct / total, per_class


def _save_fig(fig, stem):
    if SAVE_FIGS:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        p = FIGURES_DIR / f"{stem}.png"
        fig.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"Saved: {p}")


def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)


def _setup_academic_style():
    """Matplotlib rcParams for thesis / journal figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 11,
        "legend.fontsize": 12,
        "axes.titleweight": "normal",
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#2D2D2D",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "grid.color": "#C8C8C8",
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.55,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.edgecolor": "#B0B0B0",
    })


def _extract_train_loss(meta: dict, fallback=None):
    for key in ("train_loss", "train_losses"):
        if isinstance(meta, dict) and key in meta and meta[key] is not None:
            return [float(v) for v in meta[key]]
    if fallback is not None:
        return [float(v) for v in fallback]
    return None


def collect_train_losses(ann_meta, bnn_meta, tnn_train_meta=None):
    """ANN/TNN from checkpoints; BNN from checkpoint or the 07 sep run log."""
    tnn_meta = tnn_train_meta or {}
    histories = {
        "ANN": _extract_train_loss(ann_meta),
        "BNN": _extract_train_loss(bnn_meta, fallback=BNN_TRAIN_LOSS),
        "TNN": _extract_train_loss(tnn_meta),
    }
    missing = [k for k, v in histories.items() if not v]
    if missing:
        raise ValueError(f"Missing train-loss history for: {', '.join(missing)}")
    return histories


def plot_training_loss(histories: dict):
    """Overlay training loss of ANN, BNN, and TNN."""
    _setup_academic_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=200)

    for name, losses in histories.items():
        epochs = np.arange(1, len(losses) + 1)
        ax.plot(
            epochs, losses,
            color=ACADEMIC_COLORS[name],
            linewidth=2.4,
            label=name,
            marker="o",
            markersize=5.0,
            markevery=max(1, len(losses) // 10),
            markerfacecolor="white",
            markeredgewidth=1.4,
            markeredgecolor=ACADEMIC_COLORS[name],
            zorder=3,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training loss", pad=12)
    ax.set_xlim(1, max(len(v) for v in histories.values()))
    ax.set_yscale("log")
    ax.grid(True, axis="y", zorder=0, which="both")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#2D2D2D")
    ax.spines["bottom"].set_color("#2D2D2D")
    leg = ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="#B0B0B0")
    if leg:
        leg.get_frame().set_linewidth(0.6)
    fig.tight_layout()
    _save_fig(fig, "08_training_loss")
    plt.show()
    plt.close(fig)


def plot_layer_weight_summary(infos: dict):
    """
    Three horizontal panels:
      ANN — mean |weight| per layer
      BNN — % of weights at -1 and +1
      TNN — % of weights at -1, 0, and +1
    """
    _setup_academic_style()
    layers = ["fc1", "fc2", "fc3"]
    x = np.arange(len(layers))
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), dpi=200)

    # ANN: mean |weight|
    ax = axes[0]
    mean_abs = [np.abs(li["weight"]).mean() for li in infos["ANN"]]
    bars = ax.bar(x, mean_abs, color=ACADEMIC_COLORS["ANN"], edgecolor="#2D2D2D",
                  linewidth=0.65, width=0.55, zorder=3)
    for bar, v in zip(bars, mean_abs):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10.5, color="#1A1A1A")
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_ylabel(r"mean $|w|$")
    ax.set_title("ANN", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)

    # BNN: % -1 / +1
    ax = axes[1]
    width = 0.36
    bnn_neg = [(li["weight"] == -1).mean() * 100 for li in infos["BNN"]]
    bnn_pos = [(li["weight"] == 1).mean() * 100 for li in infos["BNN"]]
    ax.bar(x - width / 2, bnn_neg, width, label=r"$-1$", color=NEG_COLOR,
           edgecolor="#2D2D2D", linewidth=0.55, zorder=3)
    ax.bar(x + width / 2, bnn_pos, width, label=r"$+1$", color=POS_COLOR,
           edgecolor="#2D2D2D", linewidth=0.55, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_ylabel("% of weights")
    ax.set_ylim(0, 100)
    ax.set_title(r"BNN  $\{-1,+1\}$", pad=10)
    ax.legend(frameon=True, fancybox=False, edgecolor="#B0B0B0", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)

    # TNN: % -1 / 0 / +1
    ax = axes[2]
    width = 0.24
    tnn_neg = [(li["weight"] == -1).mean() * 100 for li in infos["TNN"]]
    tnn_zero = [(li["weight"] == 0).mean() * 100 for li in infos["TNN"]]
    tnn_pos = [(li["weight"] == 1).mean() * 100 for li in infos["TNN"]]
    ax.bar(x - width, tnn_neg, width, label=r"$-1$", color=NEG_COLOR,
           edgecolor="#2D2D2D", linewidth=0.55, zorder=3)
    ax.bar(x, tnn_zero, width, label=r"$0$", color=ZERO_COLOR,
           edgecolor="#2D2D2D", linewidth=0.55, zorder=3)
    ax.bar(x + width, tnn_pos, width, label=r"$+1$", color=POS_COLOR,
           edgecolor="#2D2D2D", linewidth=0.55, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_ylabel("% of weights")
    ax.set_ylim(0, 100)
    ax.set_title(r"TNN  $\{-1,0,+1\}$", pad=10)
    ax.legend(frameon=True, fancybox=False, edgecolor="#B0B0B0", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)

    fig.suptitle("Layer-wise weight summary", fontsize=14, y=1.03)
    fig.tight_layout()
    _save_fig(fig, "09_layer_weight_summary")
    plt.show()
    plt.close(fig)


# ============================================
# Comparison plots (ANN vs BNN vs TNN)
# ============================================
def plot_comparisons(models, infos, accs, per_class):
    names = ["ANN", "BNN", "TNN"]
    layer_names = ["fc1", "fc2", "fc3"]
    kinds = ["float", "binary", "ternary"]

    # 1) Overall test accuracy
    _setup_academic_style()
    acc_colors = ACADEMIC_COLORS
    display_labels = [
        "ANN\nfull precision",
        r"BNN" + "\n" + r"$\{-1,+1\}$",
        r"TNN" + "\n" + r"$\{-1,0,+1\}$",
    ]
    vals = [accs[n] for n in names]

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=200)
    x = np.arange(len(names))
    bars = ax.bar(
        x, vals,
        color=[acc_colors[n] for n in names],
        edgecolor="#2D2D2D",
        linewidth=0.65,
        width=0.58,
        zorder=3,
    )

    best_i = int(np.argmax(vals))
    bars[best_i].set_edgecolor("#111111")
    bars[best_i].set_linewidth(1.15)

    span = max(vals) - min(vals)
    ymin = max(0.0, min(vals) - max(8.0, span * 0.35))
    ymax = 100.0
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_locator(MultipleLocator(5 if (ymax - ymin) <= 40 else 10))
    ax.yaxis.set_minor_locator(MultipleLocator(1 if (ymax - ymin) <= 40 else 5))

    ax.set_xticks(x)
    ax.set_xticklabels(display_labels)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("MNIST test accuracy", pad=12)
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#2D2D2D")
    ax.spines["bottom"].set_color("#2D2D2D")
    ax.tick_params(axis="x", length=0)

    y_off = (ymax - ymin) * 0.015
    for i, (bar, v) in enumerate(zip(bars, vals)):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            v + y_off,
            f"{v:.2f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            color="#1A1A1A",
            fontweight="semibold" if i == best_i else "normal",
        )

    ax.axhline(
        accs["ANN"],
        color="#4D4D4D",
        linestyle=(0, (5, 3)),
        linewidth=1.0,
        zorder=2,
        label="ANN baseline",
    )
    leg = ax.legend(frameon=True, fancybox=False, edgecolor="#B0B0B0", loc="lower right")
    if leg:
        leg.get_frame().set_linewidth(0.6)

    fig.tight_layout()
    _save_fig(fig, "01_test_accuracy")
    plt.show()
    plt.close(fig)

    # 2) Per-class accuracy
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=FIGURE_DPI)
    digits = np.arange(10)
    width = 0.28
    for i, n in enumerate(names):
        vals = [per_class[n][d] for d in digits]
        ax.bar(digits + (i - 1) * width, vals, width, label=n, color=MODEL_COLORS[n], edgecolor="black")
    ax.set_xticks(digits)
    ax.set_xlabel("Digit")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Per-class test accuracy")
    ax.legend()
    _style_ax(ax)
    fig.tight_layout()
    _save_fig(fig, "02_per_class_accuracy")
    plt.show()
    plt.close(fig)

    # 3) Weight histograms: 3 layers × 3 models
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), dpi=FIGURE_DPI)
    for col, (n, kind) in enumerate(zip(names, kinds)):
        for row, li in enumerate(infos[n]):
            ax = axes[row, col]
            w = li["weight"].ravel()
            if kind == "ternary":
                ax.hist(w, bins=[-1.5, -0.5, 0.5, 1.5], color=MODEL_COLORS[n],
                        edgecolor="black", alpha=0.9, align="mid")
                ax.set_xticks([-1, 0, 1])
            elif kind == "binary":
                ax.hist(w, bins=[-1.5, -0.5, 0.5, 1.5], color=MODEL_COLORS[n],
                        edgecolor="black", alpha=0.9, align="mid")
                ax.set_xticks([-1, 1])
            else:
                ax.hist(w, bins=80, color=MODEL_COLORS[n], edgecolor="white", alpha=0.85)
                ax.axvline(0, color="red", linestyle="--", linewidth=1)
            if row == 0:
                ax.set_title(n)
            if col == 0:
                ax.set_ylabel(f"{li['name']}\ncount")
            ax.set_xlabel("weight")
    fig.suptitle("Weight distributions per layer", fontsize=15, y=1.01)
    fig.tight_layout()
    _save_fig(fig, "03_weight_histograms")
    plt.show()
    plt.close(fig)

    # 4) Discrete weight composition (BNN / TNN); ANN shown as n/a zeros
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), dpi=FIGURE_DPI, sharey=True)
    stack_vals = [-1.0, 0.0, 1.0]
    stack_labels = ["-1", "0", "+1"]
    stack_colors = ["#4A6FA5", "#B0B0B0", "#A63D40"]
    for ax, layer in zip(axes, layer_names):
        bottom = np.zeros(len(names))
        for v, lab, c in zip(stack_vals, stack_labels, stack_colors):
            pcts = []
            for n in names:
                w = infos[n][layer_names.index(layer)]["weight"]
                pcts.append(100.0 * (w == v).mean())
            pcts = np.array(pcts)
            ax.bar(names, pcts, bottom=bottom, label=lab, color=c, edgecolor="black", linewidth=0.4)
            bottom += pcts
        ax.set_title(layer)
        ax.set_ylim(0, 100)
        _style_ax(ax)
    axes[0].set_ylabel("Share of weights (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Weight alphabet composition  (ANN ≈ 0% exact ±1/0)", fontsize=14, y=1.12)
    fig.tight_layout()
    _save_fig(fig, "04_weight_composition")
    plt.show()
    plt.close(fig)

    # 5) Mean |weight| and sparsity (% exact zeros)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=FIGURE_DPI)
    x = np.arange(len(layer_names))
    width = 0.28
    for i, n in enumerate(names):
        mean_abs = [np.abs(li["weight"]).mean() for li in infos[n]]
        zeros = [(li["weight"] == 0).mean() * 100 for li in infos[n]]
        ax0.bar(x + (i - 1) * width, mean_abs, width, label=n, color=MODEL_COLORS[n], edgecolor="black")
        ax1.bar(x + (i - 1) * width, zeros, width, label=n, color=MODEL_COLORS[n], edgecolor="black")
    ax0.set_xticks(x)
    ax0.set_xticklabels(layer_names)
    ax0.set_ylabel("mean |weight|")
    ax0.set_title("Mean absolute weight")
    ax0.legend()
    _style_ax(ax0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layer_names)
    ax1.set_ylabel("% exact zeros")
    ax1.set_title("Sparsity (exact 0)")
    ax1.legend()
    _style_ax(ax1)
    fig.tight_layout()
    _save_fig(fig, "05_magnitude_and_sparsity")
    plt.show()
    plt.close(fig)

    # 6) Weight heatmaps: 3 models × 3 layers
    fig, axes = plt.subplots(3, 3, figsize=(14, 9), dpi=FIGURE_DPI)
    for row, (n, kind) in enumerate(zip(names, kinds)):
        for col, li in enumerate(infos[n]):
            ax = axes[row, col]
            w = li["weight"]
            if kind in ("binary", "ternary"):
                kw = dict(vmin=-1, vmax=1)
            else:
                p99 = np.percentile(np.abs(w), 99)
                kw = dict(vmin=-p99, vmax=p99)
            im = ax.imshow(w, aspect="auto", cmap="RdBu_r", **kw)
            ax.set_title(f"{n} {li['name']} {w.shape[0]}×{w.shape[1]}", fontsize=11)
            plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Weight matrices", fontsize=15)
    fig.tight_layout()
    _save_fig(fig, "06_weight_heatmaps")
    plt.show()
    plt.close(fig)

    # 7) fc1 first 8 input filters per model
    fig, axes = plt.subplots(3, 8, figsize=(14, 5.5), dpi=FIGURE_DPI)
    for row, (n, kind) in enumerate(zip(names, kinds)):
        w1 = infos[n][0]["weight"]
        for col in range(8):
            ax = axes[row, col]
            filt = w1[col].reshape(28, 28)
            if kind in ("binary", "ternary"):
                ax.imshow(filt, cmap="RdBu_r", vmin=-1, vmax=1)
            else:
                ax.imshow(filt, cmap="RdBu_r")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(n, fontsize=13)
            if row == 0:
                ax.set_title(f"n{col}", fontsize=10)
    fig.suptitle("fc1 input filters (first 8 neurons, 28×28)", fontsize=14)
    fig.tight_layout()
    _save_fig(fig, "07_fc1_filters")
    plt.show()
    plt.close(fig)

    print(f"\nComparison figures directory: {FIGURES_DIR}")


# ============================================
# Main
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("Load ANN, BNN, TNN — comparison plots")
    print("=" * 60)
    print("Pick a saved model for each network. Enter keeps the newest match.")

    ann_checkpoint = choose_checkpoint("ann")
    bnn_checkpoint = choose_checkpoint("bnn")
    tnn_checkpoint = choose_checkpoint("tnn")
    tnn_train_checkpoint = choose_checkpoint("tnn_train")

    net_ann, ann_path, ann_meta = load_ann(ann_checkpoint)
    net_bnn, bnn_path, bnn_meta = load_bnn(bnn_checkpoint)
    net_tnn, tnn_path, tnn_meta = load_tnn_ternary(tnn_checkpoint)

    test_loader = make_loaders()
    ann_acc, ann_pc = evaluate_accuracy(net_ann, test_loader)
    bnn_acc, bnn_pc = evaluate_accuracy(net_bnn, test_loader)
    tnn_acc, tnn_pc = evaluate_accuracy(net_tnn, test_loader)

    print_model_summary(net_ann, "ANN (full-precision)", ann_path, ann_meta, acc=ann_acc)
    '''
    ANN (full-precision)
      checkpoint: BASE_ANN_mnist_lr0.0001_ep100_seed42_20260908.pth
      parameters: 235,146
      saved test accuracy: 98.00%
      evaluated test accuracy: 98.00%
      layers: fc1 (256, 784), fc2 (128, 256), fc3 (10, 128)
    '''
    
    
    
    print_model_summary(net_bnn, "BNN (binary -1/+1)", bnn_path, bnn_meta, acc=bnn_acc)
    '''
    BNN (binary -1/+1)
      checkpoint: 07_sep_2026_BNN.pth
      parameters: 235,146
      saved test accuracy: n/a
      evaluated test accuracy: 37.22%
      layers: fc1 (256, 784), fc2 (128, 256), fc3 (10, 128)
    
    
    '''
    
    
    print_model_summary(net_tnn, "TNN (ternary -1/0/+1)", tnn_path, tnn_meta, acc=tnn_acc)
    '''
    TNN (ternary -1/0/+1)
      checkpoint: TERNARY_ONLY_mnist_tw0.7_th0.05_seed42_20260908.pth
      parameters: 235,146
      saved test accuracy: 93.78%
      evaluated test accuracy: 93.78%
      layers: fc1 (256, 784), fc2 (128, 256), fc3 (10, 128)
    
    
    '''

    models = {"ANN": net_ann, "BNN": net_bnn, "TNN": net_tnn}
    infos = {
        "ANN": collect_layer_info(net_ann),
        "BNN": collect_layer_info(net_bnn),
        "TNN": collect_layer_info(net_tnn),
    }
    _print_weight_stats(infos["ANN"], "ANN (full-precision)", "float")
    _print_weight_stats(infos["BNN"], "BNN (binary -1/+1)", "binary")
    _print_weight_stats(infos["TNN"], "TNN (ternary -1/0/+1)", "ternary")

    accs = {"ANN": ann_acc, "BNN": bnn_acc, "TNN": tnn_acc}
    per_class = {"ANN": ann_pc, "BNN": bnn_pc, "TNN": tnn_pc}

    tnn_train_ckpt = _load_torch_file(Path(tnn_train_checkpoint))
    loss_histories = collect_train_losses(ann_meta, bnn_meta, tnn_train_ckpt)

    print("\n" + "=" * 60)
    print("Plotting ANN vs BNN vs TNN comparisons")
    print("=" * 60)
    plot_comparisons(models, infos, accs, per_class)
    plot_training_loss(loss_histories)
    plot_layer_weight_summary(infos)

    print("\n" + "=" * 60)
    print("Done.")
    print(f"  ANN: {ann_acc:.2f}%")
    print(f"  BNN: {bnn_acc:.2f}%")
    print(f"  TNN: {tnn_acc:.2f}%")
    print("=" * 60)
