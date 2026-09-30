'''
In The Name of God

Ali Pilehvar Meibody

Last Update : 07 sep 2026


03_Base_ANN_TNN.py


In previous files (01.py and 02.py) we tried to find the best ANN and TNN
after optimization we reach the best of them and they are saved as pth


here we just after knwoing best configurations
again train ANN, TNN , Inference TNN and we save them which name like 
                                                        BASE_ANN_....
                                                        BASE_TNN_.....
                                                        TERNARY_ONLY_...
                                                        













In details:
Here we have three different things. Normal ANN, Trained TNN, ternary-only inference mdoel


ANN here is only get MNIST 784 pixel flatten , going through 3 fully-connected llayers 
and between each layers we have ReLUbut the final layer is logits for 10 numbers.

Input784 --> Linear (784-256) + ReLU _-> Linear(256-128)+ReLU -->Linear(128->10) --> 0-9 class

the weigths after learning is float/full precision like 0.0234, -0.1881 , 0.0047

ANN is our normal Baseline





TNN or ternary  neural network , has same architecture of ANN like  784-->256-->128-->10
instead of nn.Linear we used TernaryLinear, in TNN we have two weights . one is
trainable/latent/hidden which is layer.weight of pytorch and it is float and for training
we have ternary in forward that go to -1,0,+1

It means that in forward , calculations is with ternary weights.
Ternary emans that we have threshold and threshodl is 0.05
But actualy our TNN in calculation is forward ternary but itself
has latent float weights. 


What is here important? it said that in real ANN we have 
y = W*x + b which the multiple is float , summ is float and high memory , high energy and 
heavy hardware. 

But TernaryMNIST said that in forward pas used float weight , so weights just -1,0,+1
so in forward instead of 0.91*x we have 1*x or -0.72*x can be -1*x . 
so here the heavy float mutliplication rmeoved and 



Network learn how change latent weight to have better ternary forwad behavior.
this network dirctly doesnt learn float inference , but floats juist controller of state ternary

consider we have one weight w=0.01 , threshold = 0.05 so forward is 0.
but in training newtork found this connection is important, so gradiant say this
weigth must be stronger. so optimizer go
from 0.01 --> 0.03 -> 0.04 --> 0.051 and then it can cross trheshodl 


so ternary we have +1, -1 , 0 as output, and derivative is always is zero.
so gradient is zero and no learning. if no surrogate optimziare cannot learn.

Float weigths is improtant not for inference, for possible trianign and state ternary




The mdoel is not real teranry, because we have latent float, and memory must 
keep floats. so for hardware we just want ternary weights. so we have TernaryInferenceMNIST

It is Pure inference mdoel. this is register_buffer, means that 
it is not tranable tensor, no optimzier, no gradient, no update , just for inference
just isnide we have -1,0,+1
how we create them --> with build_ternary_inference_model(...)

we get ternary final from trained mdoel.
so TernaryInferenceMNIST remove latent wiegth,s surrogate gradient,tthresholding logic
autiograd logic. 
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


from datetime import date
from pathlib import Path

import torch.optim as optim

from paths import DATA_DIR, FIGURES_DIR as PROJECT_FIGURES_DIR, PTH_DIR


# ============================================
'''                Variables              '''
# ============================================
SEED = 42  # change this to try other runs; keep fixed for identical results

batch_size = 16
BATCH_SIZE = batch_size

#learning_rate = 0.0001 #for ANN
learning_rate=1e-4 #for TNN
LEARNING_RATE = learning_rate

num_epochs = 100
NUM_EPOCHS = num_epochs

SAVE_FIGS = False
SAVE_CHECKPOINTS = True

FIGURE_DPI = 150
THRESHOLD = 0.05
SMOOTH_TW_WIDTH = 0.7
DEVICE = torch.device("cpu")

FIGURES_DIR = PROJECT_FIGURES_DIR / "base_ann_tnn"
TERNARY_INFERENCE_FIGURES_DIR = PROJECT_FIGURES_DIR / "tnn_ternary_inference_only"



# ============================================
# 0) Reproducibility — fix all random sources
# ============================================

def set_seed(seed: int = SEED) -> None:
    """Set seeds so weight init, shuffling, and training are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuDNN / CUDA: deterministic ops (relevant if you switch to GPU later)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch >= 1.8: stricter determinism on CUDA (no effect on pure CPU)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # Fail on known non-deterministic ops (use warn_only=True if something breaks)
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_loaders():
    """MNIST loaders with fixed shuffle order via Generator."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    train_dataset = datasets.MNIST(
        root=str(DATA_DIR), train=True, transform=transform, download=True,
    )
    test_dataset = datasets.MNIST(
        root=str(DATA_DIR), train=False, transform=transform, download=True,
    )
    train_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=train_generator,
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader




# ============================================
# ANN
# ============================================
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
    
    
# ============================================
# TNN — smooth threshold-window surrogate (w=0.7)
# ============================================
class TernaryWeightFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, threshold, width):
        ctx.save_for_backward(weight)
        ctx.threshold = threshold
        ctx.width = width
        return torch.where(
            weight > threshold,
            torch.ones_like(weight),
            torch.where(weight < -threshold, -torch.ones_like(weight), torch.zeros_like(weight)),
        )

    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        threshold = ctx.threshold
        width = ctx.width
        dist_pos = (weight - threshold).abs()
        dist_neg = (weight + threshold).abs()
        grad_pos = torch.clamp(1.0 - dist_pos / width, min=0.0)
        grad_neg = torch.clamp(1.0 - dist_neg / width, min=0.0)
        grad_surrogate = torch.maximum(grad_pos, grad_neg)
        return grad_output * grad_surrogate, None, None


class TernaryLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, threshold=THRESHOLD):
        super().__init__(in_features, out_features, bias)
        self.threshold = threshold

    def ternary_weight(self):
        return TernaryWeightFunction.apply(self.weight, self.threshold, SMOOTH_TW_WIDTH)

    def forward(self, input):
        return F.linear(input, self.ternary_weight(), self.bias)


class TernaryMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = TernaryLinear(784, 256, threshold=THRESHOLD)
        self.fc2 = TernaryLinear(256, 128, threshold=THRESHOLD)
        self.fc3 = TernaryLinear(128, 10, threshold=THRESHOLD)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ============================================
# Ternary-only inference model (weights frozen at -1, 0, +1)
# ============================================
class TernaryInferenceLinear(nn.Module):
    """Inference layer: buffers hold only quantized {-1, 0, +1} weights."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.zeros(out_features, in_features))
        self.register_buffer("bias", torch.zeros(out_features))

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)


class TernaryInferenceMNIST(nn.Module):
    """Deployable TNN: no latent weights, only ternary matrices + biases."""

    def __init__(self):
        super().__init__()
        self.fc1 = TernaryInferenceLinear(784, 256)
        self.fc2 = TernaryInferenceLinear(256, 128)
        self.fc3 = TernaryInferenceLinear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    
    

# ============================================
# Train / evaluate
# ============================================

#This model used for ANN and TNN
def train_model(model, train_loader, criterion, optimizer, num_epochs=NUM_EPOCHS):
    history = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        history.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
    return history


#Used for ANN and TNN 
def evaluate_accuracy(model, test_loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total





#This is used for TNN and ANN
def collect_linear_layer_info(model):
    """Layer stats from nn.Linear or TernaryLinear (full-precision weights)."""
    layer_info = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, TernaryLinear)):
            w = module.weight.data.cpu().numpy()
            b = module.bias.data.cpu().numpy()
            n_w, n_b = w.size, b.size
            layer_info.append({
                "name": name,
                "in_features": module.in_features,
                "out_features": module.out_features,
                "weight_shape": tuple(w.shape),
                "bias_shape": tuple(b.shape),
                "n_weights": n_w,
                "n_bias": n_b,
                "n_total": n_w + n_b,
                "weight": w,
                "bias": b,
            })
    return layer_info


#This is used for TernaryInferenceMNIST
def collect_ternary_layer_info(model: TernaryInferenceMNIST):
    """Layer stats from ternary inference buffers only."""
    layer_info = []
    for name in ("fc1", "fc2", "fc3"):
        layer = getattr(model, name)
        w = layer.weight.cpu().numpy()
        b = layer.bias.cpu().numpy()
        n_w, n_b = w.size, b.size
        layer_info.append({
            "name": name,
            "in_features": layer.in_features,
            "out_features": layer.out_features,
            "weight_shape": tuple(w.shape),
            "bias_shape": tuple(b.shape),
            "n_weights": n_w,
            "n_bias": n_b,
            "n_total": n_w + n_b,
            "weight": w,
            "bias": b,
        })
    return layer_info








#This is used for any info that comes from collect_linear_layer_info() for ANN,TNN
#and also collect_inference_ternary_layer_info() for Inference TNN

def print_layer_summary(layer_info, model_label):
    total_params = sum(li["n_total"] for li in layer_info)
    print(f"\n{'=' * 60}")
    print(f"{model_label} — parameters: {total_params:,}")
    print("-" * 60)
    print(f"{'Layer':<8} {'Shape (W)':<16} {'#W':>10} {'#b':>8} {'Total':>10}")
    print("-" * 60)
    for li in layer_info:
        print(
            f"{li['name']:<8} {str(li['weight_shape']):<16} {li['n_weights']:>10,} "
            f"{li['n_bias']:>8,} {li['n_total']:>10,}"
        )
    print("-" * 60)







def activation_snapshot(model, test_loader, is_tnn=False):
    model.eval()
    with torch.no_grad():
        sample_images, _ = next(iter(test_loader))
        x = sample_images.to(DEVICE)
        if is_tnn:
            h1 = F.relu(model.fc1(x))
            h2 = F.relu(model.fc2(h1))
            logits = model.fc3(h2)
        else:
            h1 = model.relu(model.fc1(x))
            h2 = model.relu(model.fc2(h1))
            logits = model.fc3(h2)
    return {
        "input (flattened)": x.cpu().numpy(),
        "after fc1+ReLU": h1.cpu().numpy(),
        "after fc2+ReLU": h2.cpu().numpy(),
        "logits (fc3)": logits.cpu().numpy(),
    }



def activation_snapshot_ternary_inference(model, test_loader):
    model.eval()
    with torch.no_grad():
        sample_images, _ = next(iter(test_loader))
        x = sample_images.to(DEVICE)
        h1 = F.relu(model.fc1(x))
        h2 = F.relu(model.fc2(h1))
        logits = model.fc3(h2)
    return {
        "input (flattened)": x.cpu().numpy(),
        "after fc1+ReLU": h1.cpu().numpy(),
        "after fc2+ReLU": h2.cpu().numpy(),
        "logits (fc3)": logits.cpu().numpy(),
    }











# ============================================
# Shared EDA plots (same layout for ANN and TNN)
# ============================================
def run_eda_plots(layer_info, activations, prefix, title_tag):
    """prefix: e.g. 'ann' or 'tnn' — used in filenames."""
    n_layers = len(layer_info)
    names = [li["name"] for li in layer_info]
    std_w = [li["weight"].std() for li in layer_info]
    mean_abs_w = [np.abs(li["weight"]).mean() for li in layer_info]
    mean_abs_b = [np.abs(li["bias"]).mean() for li in layer_info]

    # 1) Per-layer weight & bias histograms
    fig, axes = plt.subplots(n_layers, 2, figsize=(12, 4 * n_layers))
    if n_layers == 1:
        axes = np.array([axes])
    for row, li in enumerate(layer_info):
        w, b = li["weight"].ravel(), li["bias"].ravel()
        axes[row, 0].hist(w, bins=80, color="steelblue", edgecolor="white", alpha=0.85)
        axes[row, 0].axvline(0, color="red", linestyle="--", linewidth=1, label="zero")
        axes[row, 0].set_title(f"{li['name']} — weight distribution (n={len(w):,})")
        axes[row, 0].set_xlabel("weight value")
        axes[row, 0].set_ylabel("count")
        axes[row, 0].legend()
        axes[row, 1].hist(b, bins=40, color="darkorange", edgecolor="white", alpha=0.85)
        axes[row, 1].axvline(0, color="red", linestyle="--", linewidth=1, label="zero")
        axes[row, 1].set_title(f"{li['name']} — bias distribution (n={len(b):,})")
        axes[row, 1].set_xlabel("bias value")
        axes[row, 1].set_ylabel("count")
        axes[row, 1].legend()
    plt.suptitle(f"{title_tag} — per-layer weight & bias histograms", fontsize=13, y=1.01)
    plt.tight_layout()
    p = FIGURES_DIR / f"{prefix}_eda_layer_distributions.png"
    if SAVE_FIGS:
        plt.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.show()
    
    plt.close()

    # 2) Mean |weight| and |bias| per layer
    fig2, ax2 = plt.subplots(1, 2, figsize=(10, 4))
    ax2[0].bar(names, mean_abs_w, color="steelblue", edgecolor="black")
    ax2[0].set_title("Mean |weight| per layer")
    ax2[0].set_ylabel("mean |w|")
    ax2[1].bar(names, mean_abs_b, color="darkorange", edgecolor="black")
    ax2[1].set_title("Mean |bias| per layer")
    ax2[1].set_ylabel("mean |b|")
    plt.suptitle(f"{title_tag} — average magnitude per layer", fontsize=12)
    plt.tight_layout()
    p = FIGURES_DIR / f"{prefix}_eda_layer_magnitudes.png"
    if SAVE_FIGS:
        plt.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.show()
    plt.close()

    # 3) Weight std per layer
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.bar(names, std_w, color="seagreen", edgecolor="black")
    ax3.set_title("Weight standard deviation per layer")
    ax3.set_ylabel("std(weights)")
    plt.tight_layout()
    p = FIGURES_DIR / f"{prefix}_eda_weight_std_per_layer.png"
    if SAVE_FIGS:
        plt.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.show()
    plt.close()

    # 4) Heatmaps fc3 + fc1 subset
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
    w3 = layer_info[-1]["weight"]
    im0 = axes4[0].imshow(
        w3, aspect="auto", cmap="RdBu_r",
        vmin=-np.percentile(np.abs(w3), 99), vmax=np.percentile(np.abs(w3), 99),
    )
    axes4[0].set_title("fc3 weight matrix (10 outputs × 128 inputs)")
    axes4[0].set_xlabel("input neuron (hidden2)")
    axes4[0].set_ylabel("output class (0–9)")
    plt.colorbar(im0, ax=axes4[0], fraction=0.046)
    w1 = layer_info[0]["weight"][:32, :]
    im1 = axes4[1].imshow(
        w1, aspect="auto", cmap="RdBu_r",
        vmin=-np.percentile(np.abs(w1), 99), vmax=np.percentile(np.abs(w1), 99),
    )
    axes4[1].set_title("fc1 weights — first 32 neurons × 784 pixels")
    axes4[1].set_xlabel("flattened pixel index")
    axes4[1].set_ylabel("hidden neuron (first 32)")
    plt.colorbar(im1, ax=axes4[1], fraction=0.046)
    plt.suptitle(f"{title_tag} — weight heatmaps", fontsize=12)
    plt.tight_layout()
    p = FIGURES_DIR / f"{prefix}_eda_weight_heatmaps.png"
    if SAVE_FIGS:
        plt.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.show()
    plt.close()

    # 5) Activation distributions
    fig5, axes5 = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (act_name, arr) in zip(axes5.ravel(), activations.items()):
        flat = arr.ravel()
        ax.hist(flat, bins=80, color="purple", alpha=0.75, edgecolor="white")
        ax.set_title(f"{act_name}\n(mean={flat.mean():.3f}, std={flat.std():.3f})")
        ax.set_xlabel("activation value")
        ax.set_ylabel("count")
    plt.suptitle(f"{title_tag} — activation distributions (one test batch)", fontsize=12)
    plt.tight_layout()
    p = FIGURES_DIR / f"{prefix}_eda_activation_distributions.png"
    if SAVE_FIGS:
        plt.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.show()
    plt.close()

    # 6) fc1 filters (first 16 neurons)
    fig6, axes6 = plt.subplots(4, 4, figsize=(8, 8))
    w1_full = layer_info[0]["weight"]
    for i, ax in enumerate(axes6.ravel()):
        ax.imshow(w1_full[i].reshape(28, 28), cmap="RdBu_r")
        ax.set_title(f"fc1 neuron {i}", fontsize=8)
        ax.axis("off")
    plt.suptitle(f"{title_tag} — fc1 first 16 learned input filters (28×28)", fontsize=12)
    plt.tight_layout()
    p = FIGURES_DIR / f"{prefix}_eda_fc1_filters.png"
    if SAVE_FIGS:
        plt.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.show()
    plt.close()


def plot_training_loss(history, prefix, title_tag):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(history) + 1), history, color="steelblue", linewidth=1.5)
    ax.set_title(f"{title_tag} — training loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = FIGURES_DIR / f"{prefix}_training_loss.png"
    if SAVE_FIGS:
        plt.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.show()
    plt.close()


def plot_tnn_ternary_extra(tnn_model, prefix, title_tag):
    """TNN-only: ternary weight heatmaps and {-1,0,+1} histograms (same color style)."""
    layers = [("fc1", tnn_model.fc1), ("fc2", tnn_model.fc2), ("fc3", tnn_model.fc3)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, layer) in zip(axes, layers):
        tw = layer.ternary_weight().detach().cpu().numpy()
        im = ax.imshow(tw, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"{name} ternary weights")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle(f"{title_tag} — ternary weight matrices", fontsize=12)
    plt.tight_layout()
    p = FIGURES_DIR / f"{prefix}_ternary_weight_heatmaps.png"
    if SAVE_FIGS:
        plt.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.show()
    plt.close()

    fig2, axes2 = plt.subplots(3, 2, figsize=(12, 10))
    for row, (name, layer) in enumerate(layers):
        full_w = layer.weight.detach().cpu().numpy().ravel()
        tern_w = layer.ternary_weight().detach().cpu().numpy().ravel()
        axes2[row, 0].hist(full_w, bins=80, color="steelblue", edgecolor="white", alpha=0.85)
        axes2[row, 0].set_title(f"{name} — latent (full-precision) weights")
        axes2[row, 1].hist(tern_w, bins=3, color="seagreen", edgecolor="black", alpha=0.85,
                           range=(-1.5, 1.5))
        axes2[row, 1].set_title(f"{name} — ternary weights (-1, 0, +1)")
    plt.suptitle(f"{title_tag} — latent vs ternary weight histograms", fontsize=12)
    plt.tight_layout()
    p = FIGURES_DIR / f"{prefix}_ternary_vs_latent_histograms.png"
    if SAVE_FIGS:
        plt.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.show()
    plt.close()

    print(f"\n{title_tag} — ternary weight counts:")
    for name, layer in layers:
        tw = layer.ternary_weight().detach()
        neg = (tw == -1).sum().item()
        zero = (tw == 0).sum().item()
        pos = (tw == 1).sum().item()
        total = tw.numel()
        print(f"  {name}: -1={neg} ({100*neg/total:.2f}%)  0={zero} ({100*zero/total:.2f}%)  "
              f"+1={pos} ({100*pos/total:.2f}%)")


def run_ternary_inference_eda_plots(layer_info, activations, title_tag="TNN inference (ternary weights only)"):
    """EDA plots using only {-1, 0, +1} weight buffers (no latent weights)."""
    out_dir = TERNARY_INFERENCE_FIGURES_DIR
    prefix = "ternary_inf"
    n_layers = len(layer_info)
    names = [li["name"] for li in layer_info]
    ternary_bins = [-1.5, -0.5, 0.5, 1.5]

    # 1) Per-layer ternary weight histograms (3 bars) + bias histograms
    fig, axes = plt.subplots(n_layers, 2, figsize=(12, 4 * n_layers))
    if n_layers == 1:
        axes = np.array([axes])
    for row, li in enumerate(layer_info):
        w, b = li["weight"].ravel(), li["bias"].ravel()
        axes[row, 0].hist(
            w, bins=ternary_bins, color="seagreen", edgecolor="black", alpha=0.9, align="mid",
        )
        axes[row, 0].set_xticks([-1, 0, 1])
        axes[row, 0].set_title(f"{li['name']} — ternary weights (n={len(w):,})")
        axes[row, 0].set_xlabel("weight value")
        axes[row, 0].set_ylabel("count")
        axes[row, 1].hist(b, bins=40, color="darkorange", edgecolor="white", alpha=0.85)
        axes[row, 1].axvline(0, color="red", linestyle="--", linewidth=1, label="zero")
        axes[row, 1].set_title(f"{li['name']} — bias distribution (n={len(b):,})")
        axes[row, 1].set_xlabel("bias value")
        axes[row, 1].legend()
    plt.suptitle(f"{title_tag} — per-layer histograms", fontsize=13, y=1.01)
    plt.tight_layout()
    p = out_dir / f"{prefix}_layer_distributions.png"
    if SAVE_FIGS:
        plt.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.show()
    plt.close()

    # 2) Ternary counts bar chart per layer
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    x = np.arange(len(names))
    width = 0.25
    neg_pct = [(li["weight"] == -1).mean() * 100 for li in layer_info]
    zero_pct = [(li["weight"] == 0).mean() * 100 for li in layer_info]
    pos_pct = [(li["weight"] == 1).mean() * 100 for li in layer_info]
    ax2.bar(x - width, neg_pct, width, label="-1", color="#0C5DA5")
    ax2.bar(x, zero_pct, width, label="0", color="#888888")
    ax2.bar(x + width, pos_pct, width, label="+1", color="#1B7F3B")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylabel("% of weights")
    ax2.set_title(f"{title_tag} — ternary composition per layer")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    p = out_dir / f"{prefix}_composition_per_layer.png"
    if SAVE_FIGS:
        plt.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.show()
    plt.close()

    # 3) Heatmaps (ternary matrices)
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    w3 = layer_info[-1]["weight"]
    im0 = axes3[0].imshow(w3, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    axes3[0].set_title("fc3 ternary weights (10 × 128)")
    axes3[0].set_xlabel("input neuron")
    axes3[0].set_ylabel("output class")
    plt.colorbar(im0, ax=axes3[0], fraction=0.046, ticks=[-1, 0, 1])
    w1 = layer_info[0]["weight"][:32, :]
    im1 = axes3[1].imshow(w1, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    axes3[1].set_title("fc1 ternary — first 32 neurons × 784")
    axes3[1].set_xlabel("pixel index")
    axes3[1].set_ylabel("neuron")
    plt.colorbar(im1, ax=axes3[1], fraction=0.046, ticks=[-1, 0, 1])
    plt.suptitle(f"{title_tag} — ternary weight heatmaps", fontsize=12)
    plt.tight_layout()
    p = out_dir / f"{prefix}_weight_heatmaps.png"
    if SAVE_FIGS:
        plt.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.show()
    plt.close()

    # 4) Full fc1, fc2, fc3 ternary heatmaps
    fig4, axes4 = plt.subplots(1, 3, figsize=(15, 4))
    for ax, li in zip(axes4, layer_info):
        im = ax.imshow(li["weight"], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"{li['name']} ({li['weight_shape'][0]}×{li['weight_shape'][1]})")
        plt.colorbar(im, ax=ax, fraction=0.046, ticks=[-1, 0, 1])
    plt.suptitle(f"{title_tag} — all ternary weight matrices", fontsize=12)
    plt.tight_layout()
    p = out_dir / f"{prefix}_all_layer_heatmaps.png"
    if SAVE_FIGS:
        plt.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.show()
    plt.close()

    # 5) Activation distributions
    fig5, axes5 = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (act_name, arr) in zip(axes5.ravel(), activations.items()):
        flat = arr.ravel()
        ax.hist(flat, bins=80, color="purple", alpha=0.75, edgecolor="white")
        ax.set_title(f"{act_name}\n(mean={flat.mean():.3f}, std={flat.std():.3f})")
        ax.set_xlabel("activation")
        ax.set_ylabel("count")
    plt.suptitle(f"{title_tag} — activations (one test batch)", fontsize=12)
    plt.tight_layout()
    p = out_dir / f"{prefix}_activation_distributions.png"
    if SAVE_FIGS:
        plt.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.show()
    plt.close()

    # 6) fc1 ternary filters (first 16 neurons as 28×28)
    fig6, axes6 = plt.subplots(4, 4, figsize=(8, 8))
    w1_full = layer_info[0]["weight"]
    for i, ax in enumerate(axes6.ravel()):
        ax.imshow(w1_full[i].reshape(28, 28), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"fc1 neuron {i}", fontsize=8)
        ax.axis("off")
    plt.suptitle(f"{title_tag} — fc1 ternary input filters (28×28)", fontsize=12)
    plt.tight_layout()
    p = out_dir / f"{prefix}_fc1_filters.png"
    if SAVE_FIGS:
        plt.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.show()
    plt.close()











def save_checkpoint(model, filename, extra=None):
    payload = {
        "state_dict": model.state_dict(),
        "seed": SEED,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
    }
    if extra:
        payload.update(extra)
    path = PTH_DIR / filename
    torch.save(payload, path)
    print(f"Saved checkpoint: {path}")
    return path


def save_ternary_only_checkpoint(model, filename, extra=None):
    payload = {
        "state_dict": model.state_dict(),
        "weight_type": "ternary_inference_only",
        "values": "[-1, 0, +1]",
        "threshold_used_for_quantization": THRESHOLD,
        "seed": SEED,
    }
    if extra:
        payload.update(extra)
    path = PTH_DIR / filename
    torch.save(payload, path)
    print(f"Saved ternary-only checkpoint: {path}")
    return path



# ============================================
# ============================================
# ============================================
# ============================================
# ============================================
#                     Main
# ============================================
# ============================================
# ============================================
# ============================================
# ============================================

set_seed(SEED)
print(f"Random seed: {SEED} (reproducible run)")
print(f"Config: epochs={NUM_EPOCHS}, lr={LEARNING_RATE}, batch={BATCH_SIZE}")
print(f"TNN surrogate: smooth threshold-window, width={SMOOTH_TW_WIDTH}")

train_loader, test_loader = make_loaders()
criterion = nn.CrossEntropyLoss()
run_date = date.today().isoformat().replace("-", "")


#=====================
#=====================
#=====================
#=====================
#=====================
#=====================
# ----- ANN -----
#=====================
#=====================
#=====================
#=====================
#=====================
#=====================

print("\n" + "=" * 60)
print("Training BASE ANN")
print("=" * 60)
set_seed(SEED)

ann_model = ANN().to(DEVICE)
ann_optimizer = optim.Adam(ann_model.parameters(), lr=LEARNING_RATE)
ann_history = train_model(ann_model, train_loader, criterion, ann_optimizer)


'''
07 sep


Epoch [1/100], Loss: 0.3546
Epoch [2/100], Loss: 0.1604
Epoch [3/100], Loss: 0.1100
Epoch [4/100], Loss: 0.0816
Epoch [5/100], Loss: 0.0636
Epoch [6/100], Loss: 0.0501
Epoch [7/100], Loss: 0.0403
Epoch [8/100], Loss: 0.0325
Epoch [9/100], Loss: 0.0262
Epoch [10/100], Loss: 0.0214
Epoch [11/100], Loss: 0.0172
Epoch [12/100], Loss: 0.0141
Epoch [13/100], Loss: 0.0118
Epoch [14/100], Loss: 0.0097
Epoch [15/100], Loss: 0.0086
Epoch [16/100], Loss: 0.0074
Epoch [17/100], Loss: 0.0062
Epoch [18/100], Loss: 0.0053
Epoch [19/100], Loss: 0.0056
Epoch [20/100], Loss: 0.0041
Epoch [21/100], Loss: 0.0046
Epoch [22/100], Loss: 0.0037
Epoch [23/100], Loss: 0.0033
Epoch [24/100], Loss: 0.0033
Epoch [25/100], Loss: 0.0033
Epoch [26/100], Loss: 0.0029
Epoch [27/100], Loss: 0.0026
Epoch [28/100], Loss: 0.0033
Epoch [29/100], Loss: 0.0027
Epoch [30/100], Loss: 0.0023
Epoch [31/100], Loss: 0.0029
Epoch [32/100], Loss: 0.0018
Epoch [33/100], Loss: 0.0026
Epoch [34/100], Loss: 0.0017
Epoch [35/100], Loss: 0.0027
Epoch [36/100], Loss: 0.0027
Epoch [37/100], Loss: 0.0014
Epoch [38/100], Loss: 0.0021
Epoch [39/100], Loss: 0.0016
Epoch [40/100], Loss: 0.0022
Epoch [41/100], Loss: 0.0021
Epoch [42/100], Loss: 0.0024
Epoch [43/100], Loss: 0.0014
Epoch [44/100], Loss: 0.0023
Epoch [45/100], Loss: 0.0015
Epoch [46/100], Loss: 0.0018
Epoch [47/100], Loss: 0.0016
Epoch [48/100], Loss: 0.0020
Epoch [49/100], Loss: 0.0016
Epoch [50/100], Loss: 0.0020
Epoch [51/100], Loss: 0.0016
Epoch [52/100], Loss: 0.0017
Epoch [53/100], Loss: 0.0014
Epoch [54/100], Loss: 0.0002
Epoch [55/100], Loss: 0.0026
Epoch [56/100], Loss: 0.0017
Epoch [57/100], Loss: 0.0019
Epoch [58/100], Loss: 0.0014
Epoch [59/100], Loss: 0.0017
Epoch [60/100], Loss: 0.0021
Epoch [61/100], Loss: 0.0013
Epoch [62/100], Loss: 0.0023
Epoch [63/100], Loss: 0.0004
Epoch [64/100], Loss: 0.0018
Epoch [65/100], Loss: 0.0016
Epoch [66/100], Loss: 0.0023
Epoch [67/100], Loss: 0.0013
Epoch [68/100], Loss: 0.0012
Epoch [69/100], Loss: 0.0021
Epoch [70/100], Loss: 0.0015
Epoch [71/100], Loss: 0.0011
Epoch [72/100], Loss: 0.0016
Epoch [73/100], Loss: 0.0013
Epoch [74/100], Loss: 0.0010
Epoch [75/100], Loss: 0.0013
Epoch [76/100], Loss: 0.0017
Epoch [77/100], Loss: 0.0008
Epoch [78/100], Loss: 0.0018
Epoch [79/100], Loss: 0.0006
Epoch [80/100], Loss: 0.0012
Epoch [81/100], Loss: 0.0015
Epoch [82/100], Loss: 0.0015
Epoch [83/100], Loss: 0.0007
Epoch [84/100], Loss: 0.0011
Epoch [85/100], Loss: 0.0010
Epoch [86/100], Loss: 0.0020
Epoch [87/100], Loss: 0.0006
Epoch [88/100], Loss: 0.0012
Epoch [89/100], Loss: 0.0010
Epoch [90/100], Loss: 0.0019
Epoch [91/100], Loss: 0.0013
Epoch [92/100], Loss: 0.0012
Epoch [93/100], Loss: 0.0013
Epoch [94/100], Loss: 0.0015
Epoch [95/100], Loss: 0.0006
Epoch [96/100], Loss: 0.0018
Epoch [97/100], Loss: 0.0013
Epoch [98/100], Loss: 0.0011
Epoch [99/100], Loss: 0.0014
Epoch [100/100], Loss: 0.0010


'''







ann_acc = evaluate_accuracy(ann_model, test_loader)
print(f"ANN Test Accuracy: {ann_acc:.2f}%")
#ANN Test Accuracy: 98.00%


ann_layer_info = collect_linear_layer_info(ann_model)
print_layer_summary(ann_layer_info, "ANN")
'''
============================================================
ANN — parameters: 235,146
------------------------------------------------------------
Layer    Shape (W)                #W       #b      Total
------------------------------------------------------------
fc1      (256, 784)          200,704      256    200,960
fc2      (128, 256)           32,768      128     32,896
fc3      (10, 128)             1,280       10      1,290
------------------------------------------------------------

exactly same with 01_First_ANN.py
'''

ann_activations = activation_snapshot(ann_model, test_loader, is_tnn=False)
run_eda_plots(ann_layer_info, ann_activations, "ann", "MNIST ANN")
plot_training_loss(ann_history, "ann", "MNIST ANN")

if SAVE_CHECKPOINTS:
    save_checkpoint(
        ann_model,
        f"BASE_ANN_mnist_lr{LEARNING_RATE}_ep{NUM_EPOCHS}_seed{SEED}_{run_date}.pth",
        extra={"model_type": "ANN", "test_accuracy": ann_acc, "train_loss": ann_history},
    )










#=====================
#=====================
#=====================
#=====================
#=====================
#=====================
# ----- TNN -----
#=====================
#=====================
#=====================
#=====================
#=====================
#=====================

print("\n" + "=" * 60)
print("Training TNN (smooth threshold-window, w=0.7)")
print("=" * 60)
set_seed(SEED)  # same seed → comparable fresh init, reproducible
train_loader, test_loader = make_loaders()
tnn_model = TernaryMNIST().to(DEVICE)

criterion = nn.CrossEntropyLoss()

tnn_optimizer = optim.Adam(tnn_model.parameters(), lr=LEARNING_RATE)
tnn_history = train_model(tnn_model, train_loader, criterion, tnn_optimizer)

'''
07 sep 2026


============================================================
Training TNN (smooth threshold-window, w=0.7)
============================================================
Epoch [1/100], Loss: 1.6787
Epoch [2/100], Loss: 0.8369
Epoch [3/100], Loss: 0.7047
Epoch [4/100], Loss: 0.6094
Epoch [5/100], Loss: 0.5835
Epoch [6/100], Loss: 0.5500
Epoch [7/100], Loss: 0.5503
Epoch [8/100], Loss: 0.5546
Epoch [9/100], Loss: 0.5275
Epoch [10/100], Loss: 0.5022
Epoch [11/100], Loss: 0.4926
Epoch [12/100], Loss: 0.4749
Epoch [13/100], Loss: 0.4572
Epoch [14/100], Loss: 0.4480
Epoch [15/100], Loss: 0.4349
Epoch [16/100], Loss: 0.4588
Epoch [17/100], Loss: 0.4432
Epoch [18/100], Loss: 0.4275
Epoch [19/100], Loss: 0.4190
Epoch [20/100], Loss: 0.4139
Epoch [21/100], Loss: 0.4289
Epoch [22/100], Loss: 0.4162
Epoch [23/100], Loss: 0.4057
Epoch [24/100], Loss: 0.4331
Epoch [25/100], Loss: 0.4202
Epoch [26/100], Loss: 0.3936
Epoch [27/100], Loss: 0.4066
Epoch [28/100], Loss: 0.4038
Epoch [29/100], Loss: 0.4102
Epoch [30/100], Loss: 0.3887
Epoch [31/100], Loss: 0.3948
Epoch [32/100], Loss: 0.3842
Epoch [33/100], Loss: 0.3910
Epoch [34/100], Loss: 0.3649
Epoch [35/100], Loss: 0.3728
Epoch [36/100], Loss: 0.3687
Epoch [37/100], Loss: 0.3596
Epoch [38/100], Loss: 0.3665
Epoch [39/100], Loss: 0.3632
Epoch [40/100], Loss: 0.3481
Epoch [41/100], Loss: 0.3457
Epoch [42/100], Loss: 0.3496
Epoch [43/100], Loss: 0.3410
Epoch [44/100], Loss: 0.3370
Epoch [45/100], Loss: 0.3307
Epoch [46/100], Loss: 0.3280
Epoch [47/100], Loss: 0.3347
Epoch [48/100], Loss: 0.3154
Epoch [49/100], Loss: 0.3324
Epoch [50/100], Loss: 0.3295
Epoch [51/100], Loss: 0.3257
Epoch [52/100], Loss: 0.3197
Epoch [53/100], Loss: 0.3144
Epoch [54/100], Loss: 0.3060
Epoch [55/100], Loss: 0.3094
Epoch [56/100], Loss: 0.3364
Epoch [57/100], Loss: 0.3099
Epoch [58/100], Loss: 0.3155
Epoch [59/100], Loss: 0.3227
Epoch [60/100], Loss: 0.3276
Epoch [61/100], Loss: 0.3051
Epoch [62/100], Loss: 0.3117
Epoch [63/100], Loss: 0.3098
Epoch [64/100], Loss: 0.3205
Epoch [65/100], Loss: 0.3104
Epoch [66/100], Loss: 0.3059
Epoch [67/100], Loss: 0.3055
Epoch [68/100], Loss: 0.2963
Epoch [69/100], Loss: 0.3077
Epoch [70/100], Loss: 0.3048
Epoch [71/100], Loss: 0.2929
Epoch [72/100], Loss: 0.2681
Epoch [73/100], Loss: 0.2715
Epoch [74/100], Loss: 0.3084
Epoch [75/100], Loss: 0.3023
Epoch [76/100], Loss: 0.2960
Epoch [77/100], Loss: 0.2993
Epoch [78/100], Loss: 0.2925
Epoch [79/100], Loss: 0.2886
Epoch [80/100], Loss: 0.2902
Epoch [81/100], Loss: 0.2977
Epoch [82/100], Loss: 0.2682
Epoch [83/100], Loss: 0.2740
Epoch [84/100], Loss: 0.2870
Epoch [85/100], Loss: 0.2848
Epoch [86/100], Loss: 0.2802
Epoch [87/100], Loss: 0.2845
Epoch [88/100], Loss: 0.2972
Epoch [89/100], Loss: 0.2942
Epoch [90/100], Loss: 0.2794
Epoch [91/100], Loss: 0.2634
Epoch [92/100], Loss: 0.2721
Epoch [93/100], Loss: 0.2648
Epoch [94/100], Loss: 0.2566
Epoch [95/100], Loss: 0.2735
Epoch [96/100], Loss: 0.2547
Epoch [97/100], Loss: 0.2427
Epoch [98/100], Loss: 0.2645
Epoch [99/100], Loss: 0.2639
Epoch [100/100], Loss: 0.2468

'''

tnn_acc = evaluate_accuracy(tnn_model, test_loader)
print(f"TNN Test Accuracy: {tnn_acc:.2f}%")
#TNN Test Accuracy: 93.78%

tnn_layer_info = collect_linear_layer_info(tnn_model)
print_layer_summary(tnn_layer_info, "TNN")
'''
============================================================
TNN — parameters: 235,146
------------------------------------------------------------
Layer    Shape (W)                #W       #b      Total
------------------------------------------------------------
fc1      (256, 784)          200,704      256    200,960
fc2      (128, 256)           32,768      128     32,896
fc3      (10, 128)             1,280       10      1,290
------------------------------------------------------------

'''
tnn_activations = activation_snapshot(tnn_model, test_loader, is_tnn=True)
run_eda_plots(tnn_layer_info, tnn_activations, "tnn", "MNIST TNN (smooth TW 0.7)")
plot_training_loss(tnn_history, "tnn", "MNIST TNN (smooth TW 0.7)")
plot_tnn_ternary_extra(tnn_model, "tnn", "MNIST TNN")

if SAVE_CHECKPOINTS:
    save_checkpoint(
        tnn_model,
        f"BASE_TNN_mnist_smooth_tw_w{SMOOTH_TW_WIDTH}_lr{LEARNING_RATE}_ep{NUM_EPOCHS}_seed{SEED}_{run_date}.pth",
        extra={
            "model_type": "TNN",
            "surrogate": "smooth_threshold_window",
            "width": SMOOTH_TW_WIDTH,
            "threshold": THRESHOLD,
            "test_accuracy": tnn_acc,
            "train_loss": tnn_history,
        },
    )
























#=====================
#=====================
#=====================
#=====================
#=====================
#=====================
# --- Comparison ---
#=====================
#=====================
#=====================
#=====================
#=====================
#=====================



def plot_comparison(ann_history, tnn_history, ann_acc, tnn_acc):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, NUM_EPOCHS + 1)
    axes[0].plot(epochs, ann_history, label="ANN", color="#0C5DA5", linewidth=1.5)
    axes[0].plot(epochs, tnn_history, label="TNN (smooth TW w=0.7)", color="#1B7F3B", linewidth=1.5)
    axes[0].set_title("Training loss comparison")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    bars = axes[1].bar(["ANN", "TNN"], [ann_acc, tnn_acc], color=["#0C5DA5", "#1B7F3B"], edgecolor="black")
    axes[1].set_ylim(0, 110)
    axes[1].set_ylabel("Test accuracy (%)")
    axes[1].set_title("Test accuracy comparison")
    for bar, acc in zip(bars, [ann_acc, tnn_acc]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{acc:.2f}%", ha="center", fontsize=10)
    plt.suptitle("MNIST — ANN vs TNN (100 epochs, lr=1e-4, seed=42)", fontsize=12)
    plt.tight_layout()
    p = FIGURES_DIR / "comparison_ann_vs_tnn.png"
    if SAVE_FIGS:
        plt.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.show()
    plt.close()
    
plot_comparison(ann_history, tnn_history, ann_acc, tnn_acc)

print("\n" + "=" * 60)
print("Done.")
print(f"  ANN accuracy: {ann_acc:.2f}%")
print(f"  TNN accuracy: {tnn_acc:.2f}%")
print(f"  Figures:  {FIGURES_DIR}")
print(f"  Weights:  {PTH_DIR}")
print("=" * 60)




# ============================================
# Verify TNN weights are exactly -1, 0, +1
# ============================================
def verify_ternary_weights(model, model_label="TNN"):
    """
    Confirm forward-pass weights are exactly -1, 0, or +1 (not latent layer.weight).
    """
    allowed = {-1.0, 0.0, 1.0}
    print(f"\n{'=' * 60}")
    print(f"{model_label} — ternary weight verification (exact {{-1, 0, +1}})")
    print("=" * 60)
    all_ok = True
    for name, layer in model.named_modules():
        if not isinstance(layer, TernaryLinear):
            continue
        tw = layer.ternary_weight().detach().cpu()
        latent = layer.weight.detach().cpu()
        uniq_ternary = set(torch.unique(tw).tolist())
        uniq_latent = torch.unique(latent).numel()

        if not uniq_ternary.issubset(allowed):
            print(f"  {name}: FAIL — unexpected values: {sorted(uniq_ternary - allowed)}")
            all_ok = False
            continue
        mask = (tw == -1) | (tw == 0) | (tw == 1)
        if not mask.all():
            bad = tw[~mask]
            print(f"  {name}: FAIL — non-ternary entries, e.g. {bad[:5].tolist()}")
            all_ok = False
            continue

        n = tw.numel()
        n_neg = (tw == -1).sum().item()
        n_zero = (tw == 0).sum().item()
        n_pos = (tw == 1).sum().item()
        print(f"  {name}: OK")
        print(f"    unique ternary values: {sorted(uniq_ternary)}")
        print(f"    counts: -1={n_neg} ({100*n_neg/n:.2f}%)  "
              f"0={n_zero} ({100*n_zero/n:.2f}%)  +1={n_pos} ({100*n_pos/n:.2f}%)")
        print(f"    latent layer.weight has {uniq_latent} distinct values (not ternary)")

    if all_ok:
        print("\nAll TernaryLinear layers use exactly -1, 0, +1 in the forward pass.")
    else:
        print("\nWARNING: Some layers failed ternary verification.")
    print("=" * 60)
    return all_ok

verify_ternary_weights(tnn_model, model_label="TNN")


'''
============================================================
TNN — ternary weight verification (exact {-1, 0, +1})
============================================================
  fc1: OK
    unique ternary values: [-1.0, 0.0, 1.0]
    counts: -1=10772 (5.37%)  0=173311 (86.35%)  +1=16621 (8.28%)
    latent layer.weight has 200199 distinct values (not ternary)
  fc2: OK
    unique ternary values: [-1.0, 0.0, 1.0]
    counts: -1=5365 (16.37%)  0=25053 (76.46%)  +1=2350 (7.17%)
    latent layer.weight has 32752 distinct values (not ternary)
  fc3: OK
    unique ternary values: [-1.0, 0.0, 1.0]
    counts: -1=700 (54.69%)  0=478 (37.34%)  +1=102 (7.97%)
    latent layer.weight has 1266 distinct values (not ternary)

All TernaryLinear layers use exactly -1, 0, +1 in the forward pass.
============================================================

'''














#============================================
#============================================
#============================================
#============================================
'''      ONLY TNN WEIGHTS ARE VERIFIED      '''
#============================================
#============================================
#============================================
#============================================
#============================================
#============================================


def build_ternary_inference_model(trained_tnn: TernaryMNIST) -> TernaryInferenceMNIST:
    """Copy quantized weights from trained TNN into a ternary-only inference model."""
    inf = TernaryInferenceMNIST().to(DEVICE)
    with torch.no_grad():
        inf.fc1.weight.copy_(trained_tnn.fc1.ternary_weight())
        inf.fc1.bias.copy_(trained_tnn.fc1.bias)
        inf.fc2.weight.copy_(trained_tnn.fc2.ternary_weight())
        inf.fc2.bias.copy_(trained_tnn.fc2.bias)
        inf.fc3.weight.copy_(trained_tnn.fc3.ternary_weight())
        inf.fc3.bias.copy_(trained_tnn.fc3.bias)
    inf.eval()
    for p in inf.parameters():
        p.requires_grad = False
    return inf


def verify_ternary_inference_weights(model: TernaryInferenceMNIST, model_label="Ternary inference"):
    allowed = {-1.0, 0.0, 1.0}
    print(f"\n{'=' * 60}")
    print(f"{model_label} — stored weights (exact {{-1, 0, +1}})")
    print("=" * 60)
    all_ok = True
    for name in ("fc1", "fc2", "fc3"):
        w = getattr(model, name).weight.detach().cpu()
        uniq = set(torch.unique(w).tolist())
        if not uniq.issubset(allowed) or not ((w == -1) | (w == 0) | (w == 1)).all():
            print(f"  {name}: FAIL — values {sorted(uniq)}")
            all_ok = False
            continue
        n = w.numel()
        print(f"  {name}: OK — unique {sorted(uniq)}")
        print(f"    -1={(w == -1).sum()} ({100*(w == -1).sum()/n:.2f}%)  "
              f"0={(w == 0).sum()} ({100*(w == 0).sum()/n:.2f}%)  "
              f"+1={(w == 1).sum()} ({100*(w == 1).sum()/n:.2f}%)")
    print("=" * 60)
    return all_ok



def evaluate_ternary_on_test(model, test_loader, model_label="Ternary", save_path=None):
    """
    Full evaluation on MNIST test set (test_loader from train=False).
    Use with TernaryInferenceMNIST or TernaryMNIST (forward uses ternary weights).
    """
    model.eval()
    correct = 0
    total = 0
    class_correct = torch.zeros(10, device=DEVICE)
    class_total = torch.zeros(10, device=DEVICE)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for c in range(10):
                mask = labels == c
                class_total[c] += mask.sum()
                if mask.any():
                    class_correct[c] += (preds[mask] == labels[mask]).sum()

    accuracy = 100.0 * correct / total
    per_class = {
        int(c): 100.0 * class_correct[c].item() / class_total[c].item()
        if class_total[c] > 0 else 0.0
        for c in range(10)
    }

    print(f"\n{'-' * 60}")
    print(f"{model_label} — MNIST TEST SET (10,000 images)")
    print(f"{'-' * 60}")
    print(f"  Correct: {correct:,} / {total:,}")
    print(f"  Test accuracy (ternary weights): {accuracy:.2f}%")
    print("  Per-class accuracy (%):")
    for c in range(10):
        print(f"    digit {c}: {per_class[c]:.2f}%  "
              f"({int(class_correct[c].item())}/{int(class_total[c].item())})")
    print(f"{'-' * 60}")

    results = {
        "split": "mnist_test",
        "n_samples": total,
        "correct": correct,
        "test_accuracy_percent": accuracy,
        "per_class_accuracy_percent": per_class,
        "weight_type": "ternary",
    }

    if save_path is not None:
        save_path = Path(save_path)
        lines = [
            f"{model_label} — MNIST test evaluation",
            f"Correct: {correct} / {total}",
            f"Test accuracy: {accuracy:.4f}%",
            "",
            "Per-class accuracy (%):",
        ]
        for c in range(10):
            lines.append(f"  digit {c}: {per_class[c]:.2f}")
        save_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  Saved test report: {save_path}")

    return results


# --- Ternary-only inference: materialize {-1,0,+1}, evaluate, plot, save ---
print("\n" + "=" * 60)
print("TNN — ternary-only inference (weights in checkpoint = -1, 0, +1)")
print("=" * 60)

ternary_inf_model = build_ternary_inference_model(tnn_model)
verify_ternary_inference_weights(ternary_inf_model)

'''
============================================================
Ternary inference — stored weights (exact {-1, 0, +1})
============================================================
  fc1: OK — unique [-1.0, 0.0, 1.0]
    -1=10772 (5.37%)  0=173311 (86.35%)  +1=16621 (8.28%)
  fc2: OK — unique [-1.0, 0.0, 1.0]
    -1=5365 (16.37%)  0=25053 (76.46%)  +1=2350 (7.17%)
  fc3: OK — unique [-1.0, 0.0, 1.0]
    -1=700 (54.69%)  0=478 (37.34%)  +1=102 (7.97%)
============================================================

'''

# MNIST test set (train=False, 10k images) — ternary weights only
ternary_test_results = evaluate_ternary_on_test(
    ternary_inf_model,
    test_loader,
    model_label="Ternary-only inference",
    #save_path=TERNARY_INFERENCE_FIGURES_DIR / "ternary_test_accuracy.txt",
)

'''
------------------------------------------------------------
Ternary-only inference — MNIST TEST SET (10,000 images)
------------------------------------------------------------
  Correct: 9,378 / 10,000
  Test accuracy (ternary weights): 93.78%
  Per-class accuracy (%):
    digit 0: 94.69%  (928/980)
    digit 1: 98.68%  (1120/1135)
    digit 2: 90.50%  (934/1032)
    digit 3: 93.07%  (940/1010)
    digit 4: 94.30%  (926/982)
    digit 5: 91.59%  (817/892)
    digit 6: 95.09%  (911/958)
    digit 7: 91.34%  (939/1028)
    digit 8: 89.63%  (873/974)
    digit 9: 98.12%  (990/1009)
------------------------------------------------------------

'''
ternary_inf_acc = ternary_test_results["test_accuracy_percent"]

print(f"\nSummary:")
print(f"  Ternary-only on TEST set: {ternary_inf_acc:.2f}%")
print(f"  TNN (ternary forward) on TEST set: {tnn_acc:.2f}%")
print(f"  Difference: {abs(ternary_inf_acc - tnn_acc):.4f} percentage points")

'''
Summary:
  Ternary-only on TEST set: 93.78%
  TNN (ternary forward) on TEST set: 93.78%
  Difference: 0.0000 percentage points
  
'''

ternary_layer_info = collect_ternary_layer_info(ternary_inf_model)
print_layer_summary(ternary_layer_info, "Ternary inference")
'''
============================================================
Ternary inference — parameters: 235,146
------------------------------------------------------------
Layer    Shape (W)                #W       #b      Total
------------------------------------------------------------
fc1      (256, 784)          200,704      256    200,960
fc2      (128, 256)           32,768      128     32,896
fc3      (10, 128)             1,280       10      1,290
------------------------------------------------------------

'''
ternary_activations = activation_snapshot_ternary_inference(ternary_inf_model, test_loader)
run_ternary_inference_eda_plots(
    ternary_layer_info,
    ternary_activations,
    title_tag="MNIST TNN inference (ternary weights only)",
)

if SAVE_CHECKPOINTS:
    save_ternary_only_checkpoint(
        ternary_inf_model,
        f"TERNARY_ONLY_mnist_tw{SMOOTH_TW_WIDTH}_th{THRESHOLD}_seed{SEED}_{run_date}.pth",
        extra={
            "test_accuracy": ternary_inf_acc,
            "test_set_evaluation": ternary_test_results,
            "surrogate": "smooth_threshold_window",
            "width": SMOOTH_TW_WIDTH,
            "source_checkpoint": "latent TNN state_dict quantized at load",
        },
    )

print(f"\nTernary-only figures: {TERNARY_INFERENCE_FIGURES_DIR}")
print("Ternary-only checkpoint stores fc*.weight as {-1, 0, +1} buffers (not latent floats).")














