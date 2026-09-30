'''
In The Name of God

Ali Pilehvar Meibody

Last Update : 08 sep 2026


02b_BNN_optimization.py


One Binary Neural Network on MNIST, same architecture as the TNN:

    Input 784 --> BinaryLinear(784, 256) + ReLU
               --> BinaryLinear(256, 128) + ReLU
               --> BinaryLinear(128, 10)  --> logits

Forward weights are strictly {-1, +1} (sign of the latent float weights).
Latent floats stay trainable so the optimizer can move each weight across 0.

Saved as:
{PTH_DIR}{date_name}_BNN.pth

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

from paths import DATA_DIR, PTH_DIR





# ============================================
'''                Variables              '''
# ============================================
SEED = 42  # change this to try other runs; keep fixed for identical results
batch_size = 16
#learning_rate = 0.0001 #for ANN
learning_rate=1e-4 #for BNN

num_epochs = 100
SAVE_FIGS = False

 
date_name='07_sep_2026'


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

set_seed(SEED)
print(f"Random seed set to {SEED} (reproducible run)")


# ============================================
# 1) Transform MNIST
# ============================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.view(-1))
])

# ============================================
# 2) Load MNIST
# ============================================
train_dataset = datasets.MNIST(
    root=str(DATA_DIR),
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.MNIST(
    root=str(DATA_DIR),
    train=False,
    transform=transform,
    download=True
)


train_generator = torch.Generator().manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    generator=train_generator
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# ============================================
# 3) Binary Weight Function
# ============================================
class BinaryWeightFunction(torch.autograd.Function):
    """Forward: sign(w) in {-1, +1}. Backward: clipped STE in |w| <= 1."""

    @staticmethod
    def forward(ctx, weight):
        ctx.save_for_backward(weight)
        return torch.where(
            weight >= 0,
            torch.ones_like(weight),
            -torch.ones_like(weight)
        )

    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        grad_weight = grad_output.clone()
        grad_weight = grad_weight * (weight.abs() <= 1.0).float()
        return grad_weight


# ============================================
# 4) Binary Linear Layer
# ============================================
class BinaryLinear(nn.Linear):
    def binary_weight(self):
        return BinaryWeightFunction.apply(self.weight)

    def forward(self, input):
        return F.linear(input, self.binary_weight(), self.bias)


# ============================================
# 5) Binary MNIST Network
# ============================================
class BinaryMNIST(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = BinaryLinear(784, 256)
        self.fc2 = BinaryLinear(256, 128)
        self.fc3 = BinaryLinear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# ============================================
# 6) Model, Loss, Optimizer
# ============================================
model = BinaryMNIST()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ============================================
# 7) Training
# ============================================

train_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        if torch.isnan(outputs).any():
            print("NaN in outputs")
            break

        if torch.isnan(loss):
            print("NaN in loss")
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


'''
07 sep 2026

Epoch 1/100, Loss: 123.6093
Epoch 2/100, Loss: 14.4832
Epoch 3/100, Loss: 2.2088
Epoch 4/100, Loss: 2.2430
Epoch 5/100, Loss: 2.3112
Epoch 6/100, Loss: 2.3025
Epoch 7/100, Loss: 2.3104
Epoch 8/100, Loss: 2.3080
Epoch 9/100, Loss: 2.2952
Epoch 10/100, Loss: 2.2934
Epoch 11/100, Loss: 2.2667
Epoch 12/100, Loss: 2.2806
Epoch 13/100, Loss: 2.2753
Epoch 14/100, Loss: 2.2316
Epoch 15/100, Loss: 2.2031
Epoch 16/100, Loss: 2.2341
Epoch 17/100, Loss: 2.1799
Epoch 18/100, Loss: 2.2032
Epoch 19/100, Loss: 2.1452
Epoch 20/100, Loss: 2.2074
Epoch 21/100, Loss: 2.1752
Epoch 22/100, Loss: 2.1204
Epoch 23/100, Loss: 2.1175
Epoch 24/100, Loss: 2.1355
Epoch 25/100, Loss: 2.1435
Epoch 26/100, Loss: 2.1394
Epoch 27/100, Loss: 2.1067
Epoch 28/100, Loss: 2.1091
Epoch 29/100, Loss: 2.0654
Epoch 30/100, Loss: 2.1055
Epoch 31/100, Loss: 2.1689
Epoch 32/100, Loss: 2.0957
Epoch 33/100, Loss: 2.0080
Epoch 34/100, Loss: 2.1560
Epoch 35/100, Loss: 2.0774
Epoch 36/100, Loss: 2.1089
Epoch 37/100, Loss: 2.0778
Epoch 38/100, Loss: 2.0856
Epoch 39/100, Loss: 2.0614
Epoch 40/100, Loss: 2.0285
Epoch 41/100, Loss: 2.0287
Epoch 42/100, Loss: 2.0951
Epoch 43/100, Loss: 2.0453
Epoch 44/100, Loss: 2.0417
Epoch 45/100, Loss: 2.0754
Epoch 46/100, Loss: 2.0779
Epoch 47/100, Loss: 2.0332
Epoch 48/100, Loss: 1.9997
Epoch 49/100, Loss: 2.0183
Epoch 50/100, Loss: 1.9983
Epoch 51/100, Loss: 2.0793
Epoch 52/100, Loss: 1.9493
Epoch 53/100, Loss: 1.9592
Epoch 54/100, Loss: 2.0085
Epoch 55/100, Loss: 1.9749
Epoch 56/100, Loss: 1.9205
Epoch 57/100, Loss: 2.0008
Epoch 58/100, Loss: 1.9848
Epoch 59/100, Loss: 2.1332
Epoch 60/100, Loss: 1.9253
Epoch 61/100, Loss: 1.9505
Epoch 62/100, Loss: 1.9608
Epoch 63/100, Loss: 1.9421
Epoch 64/100, Loss: 1.9055
Epoch 65/100, Loss: 1.9259
Epoch 66/100, Loss: 1.9897
Epoch 67/100, Loss: 1.9520
Epoch 68/100, Loss: 1.9082
Epoch 69/100, Loss: 1.8763
Epoch 70/100, Loss: 1.8487
Epoch 71/100, Loss: 1.8989
Epoch 72/100, Loss: 1.9000
Epoch 73/100, Loss: 1.8708
Epoch 74/100, Loss: 1.9525
Epoch 75/100, Loss: 1.8961
Epoch 76/100, Loss: 1.8729
Epoch 77/100, Loss: 1.8515
Epoch 78/100, Loss: 1.8209
Epoch 79/100, Loss: 1.8545
Epoch 80/100, Loss: 1.7986
Epoch 81/100, Loss: 1.8435
Epoch 82/100, Loss: 1.8094
Epoch 83/100, Loss: 1.7658
Epoch 84/100, Loss: 1.8275
Epoch 85/100, Loss: 1.7805
Epoch 86/100, Loss: 1.8175
Epoch 87/100, Loss: 1.7628
Epoch 88/100, Loss: 1.7571
Epoch 89/100, Loss: 1.8131
Epoch 90/100, Loss: 1.8038
Epoch 91/100, Loss: 1.7853
Epoch 92/100, Loss: 1.7986
Epoch 93/100, Loss: 1.8900
Epoch 94/100, Loss: 1.7790
Epoch 95/100, Loss: 1.7723
Epoch 96/100, Loss: 1.7348
Epoch 97/100, Loss: 1.7513
Epoch 98/100, Loss: 1.8115
Epoch 99/100, Loss: 1.8602
Epoch 100/100, Loss: 1.7714




'''



# ============================================
# 8) Evaluation
# ============================================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predicted = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
#Test Accuracy: 37.22%



#------- SAVE --------

checkpoint = {
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_loss
}

torch.save(checkpoint, PTH_DIR / f"{date_name}_BNN.pth")



# ============================================
# 9) Plot training loss (simple)
# ============================================
epochs = np.arange(1, num_epochs + 1)
layer_names = ["fc1", "fc2", "fc3"]
layers = [model.fc1, model.fc2, model.fc3]
layer_colors = ["#059669", "#7c3aed", "#dc2626"]

_plot_rc = {
    "figure.facecolor": "white",
    "axes.facecolor": "#f8fafc",
    "axes.edgecolor": "#cbd5e1",
    "axes.labelcolor": "#334155",
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "grid.color": "#e2e8f0",
    "grid.linestyle": "-",
    "font.family": "sans-serif",
}
plt.rcParams.update(_plot_rc)


def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.6)


def _layer_param_norm(module):
    """L2 norm over weight + bias of a layer."""
    sq = sum(p.detach().pow(2).sum().item() for p in module.parameters())
    return sq ** 0.5


def _layer_grad_norm(module, loss):
    """L2 norm of gradients (weight + bias); fallback if .grad is None."""
    grads = []
    for p in module.parameters():
        if not p.requires_grad:
            continue
        g = p.grad
        if g is None:
            g = torch.autograd.grad(loss, p, retain_graph=True, allow_unused=True)[0]
        if g is not None:
            grads.append(g.detach())
    if not grads:
        return 0.0
    return (sum(g.pow(2).sum().item() for g in grads)) ** 0.5


def _bar_labels(ax, bars, fmt="{:.3f}"):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h,
            fmt.format(h), ha="center", va="bottom", fontsize=9, color="#334155",
        )


# --- Figure 1: Training loss curve ---
fig1, ax1 = plt.subplots(figsize=(9, 5.5), dpi=120)
ax1.plot(
    epochs, train_losses, color="#2563eb", linewidth=2.5,
    marker="o", markersize=5, markevery=max(1, num_epochs // 10),
    markerfacecolor="white", markeredgewidth=1.5, markeredgecolor="#2563eb",
    label="Train loss", zorder=3,
)
ax1.fill_between(epochs, train_losses, min(train_losses), color="#2563eb", alpha=0.08)
ax1.set_title("Training Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-entropy loss")
ax1.set_xlim(1, num_epochs)
_style_axis(ax1)
ax1.legend(frameon=True, fancybox=True, shadow=False, edgecolor="#e2e8f0")
fig1.tight_layout()
plt.show()

# --- Figure 2: Loss change per epoch ---
loss_change = np.diff(train_losses)
epoch_change = epochs[1:]
fig2, ax2 = plt.subplots(figsize=(9, 5.5), dpi=120)
bar_colors = ["#10b981" if v <= 0 else "#ef4444" for v in loss_change]
bars2 = ax2.bar(epoch_change, loss_change, color=bar_colors, alpha=0.85, width=0.85, edgecolor="white", linewidth=0.6)
ax2.axhline(0, color="#64748b", linewidth=1.0)
ax2.set_title("Loss Change per Epoch")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Delta loss")
_style_axis(ax2)
fig2.tight_layout()
plt.show()

# --- Figure 3: Weight L2 norm per layer ---
weight_norms = [_layer_param_norm(layer) for layer in layers]
fig3, ax3 = plt.subplots(figsize=(9, 5.5), dpi=120)
bars3 = ax3.bar(layer_names, weight_norms, color=layer_colors, alpha=0.9, width=0.55, edgecolor="white", linewidth=0.8)
ax3.set_title("Full-Precision Weight Norms per Layer")
ax3.set_ylabel("L2 norm")
_style_axis(ax3)
_bar_labels(ax3, bars3, fmt="{:.2f}")
fig3.tight_layout()
plt.show()

# --- Figure 4: Gradient L2 norm per layer (one backprop step) ---
model.train()
images, labels = next(iter(train_loader))
optimizer.zero_grad(set_to_none=True)
loss = criterion(model(images), labels)
loss.backward()
grad_norms = [_layer_grad_norm(layer, loss) for layer in layers]

fig4, ax4 = plt.subplots(figsize=(9, 5.5), dpi=120)
bars4 = ax4.bar(layer_names, grad_norms, color=layer_colors, alpha=0.9, width=0.55, edgecolor="white", linewidth=0.8)
ax4.set_title("Gradient Norms per Layer (one batch)")
ax4.set_ylabel("L2 norm of gradients")
_style_axis(ax4)
_bar_labels(ax4, bars4, fmt="{:.4f}")
fig4.tight_layout()
plt.show()
