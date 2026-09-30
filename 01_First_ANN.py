'''
In The Name of God

Ali Pilehvar Meibody

Last Update : 07 Sep 2026

01_First_ANN.py

In this file , first we create ANN structure  and we saved that in 

Pth_Models/

with name mnist_ann_07_sep_2026.pth

which is our ANN reference

'''


# ============================================
'''                   Imports              '''
# ============================================
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim

from paths import DATA_DIR, PTH_DIR



# ============================================
'''                Variables              '''
# ============================================
SEED = 42  # change this to try other runs; keep fixed for identical results
batch_size = 16
device = 'cpu'
learning_rate = 0.0001
num_epochs = 100
SAVE_FIG=False

date_name ='07 sep 2026'
date_name = date_name.replace(' ','_')

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
# 1) Transformations (Flatten 28x28 imgs -> 784)
# ============================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # mean & std of MNIST
    transforms.Lambda(lambda x: x.view(-1))      # flatten 1x28x28 → 784
])

# ============================================
# 2) Load MNIST dataset
# ============================================
train_dataset = datasets.MNIST(root=str(DATA_DIR), train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root=str(DATA_DIR), train=False, transform=transform, download=True)

train_generator = torch.Generator().manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    generator=train_generator,
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ============================================
# 3) Build Fully Connected ANN Model
# ============================================
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        
        self.fc1 = nn.Linear(784, 256)     # input → hidden1
        self.fc2 = nn.Linear(256, 128)     # hidden1 → hidden2
        self.fc3 = nn.Linear(128, 10)      # hidden2 → output
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)    # logits (no softmax for CrossEntropyLoss)
        return x

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ANN().to(device)

# ============================================
# 4) Loss and Optimizer
# ============================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# ============================================
# 5) Training looop
# ============================================

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    #images, labels = images.to(device), labels.to(device)

    for images, labels in train_loader:
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

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




# ============================================
# 6) Evaluation
# ============================================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    #images, labels = images.to(device), labels.to(device)
    
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
#Test Accuracy: 98.00%





# ============================================
# 7) Exploratory Data Analysis (model weights, biases, layers)
# ============================================

print("\n" + "=" * 60)
print("MODEL ARCHITECTURE & PARAMETER OVERVIEW")
print("=" * 60)
print(model)
print()

#---------------------
total_params = 0
trainable_params = 0
layer_info = []

for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        w = module.weight.data
        b = module.bias.data
        n_w = w.numel()
        n_b = b.numel()
        n_layer = n_w + n_b
        total_params += n_layer
        trainable_params += n_layer
        layer_info.append({
            "name": name,
            "in_features": module.in_features,
            "out_features": module.out_features,
            "weight_shape": tuple(w.shape),
            "bias_shape": tuple(b.shape),
            "n_weights": n_w,
            "n_bias": n_b,
            "n_total": n_layer,
            "weight": w.cpu().numpy(),
            "bias": b.cpu().numpy(),
        })

print(f"Total parameters (all Linear layers): {total_params:,}")
print(f"  Weights: {sum(li['n_weights'] for li in layer_info):,}")
print(f"  Biases:  {sum(li['n_bias'] for li in layer_info):,}")
print()

# --- Per-layer summary table ---
print("-" * 60)
print(f"{'Layer':<8} {'Shape (W)':<16} {'#W':>10} {'#b':>8} {'Total':>10}")
print("-" * 60)
for li in layer_info:
    print(f"{li['name']:<8} {str(li['weight_shape']):<16} {li['n_weights']:>10,} "
          f"{li['n_bias']:>8,} {li['n_total']:>10,}")
print("-" * 60)




'''

============================================================
MODEL ARCHITECTURE & PARAMETER OVERVIEW
============================================================
ANN(
  (fc1): Linear(in_features=784, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=10, bias=True)
  (relu): ReLU()
)

Total parameters (all Linear layers): 235,146
  Weights: 234,752
  Biases:  394

------------------------------------------------------------
Layer    Shape (W)                #W       #b      Total
------------------------------------------------------------
fc1      (256, 784)          200,704      256    200,960
fc2      (128, 256)           32,768      128     32,896
fc3      (10, 128)             1,280       10      1,290
------------------------------------------------------------
'''








#---------------------
# --- Weight & bias statistics per layer ---
print("\n" + "=" * 60)
print("WEIGHT & BIAS STATISTICS (per layer)")
print("=" * 60)

for li in layer_info:
    w, b = li["weight"], li["bias"]
    print(f"\n--- {li['name']}  Linear({li['in_features']} -> {li['out_features']}) ---")
    print(f"  Weights: mean={w.mean():+.6f}  std={w.std():.6f}  "
          f"min={w.min():+.6f}  max={w.max():+.6f}  |w|_mean={np.abs(w).mean():.6f}")
    print(f"  Biases:  mean={b.mean():+.6f}  std={b.std():.6f}  "
          f"min={b.min():+.6f}  max={b.max():+.6f}")
    # Sparsity / near-zero share (useful for ReLU networks)
    near_zero_w = (np.abs(w) < 1e-4).mean() * 100
    near_zero_b = (np.abs(b) < 1e-4).mean() * 100
    print(f"  Near-zero (|x|<1e-4): weights {near_zero_w:.2f}%  biases {near_zero_b:.2f}%")
    


'''
============================================================
WEIGHT & BIAS STATISTICS (per layer)
============================================================

--- fc1  Linear(784 -> 256) ---
  Weights: mean=+0.001846  std=0.054868  min=-0.505663  max=+0.305334  |w|_mean=0.040155
  Biases:  mean=-0.005622  std=0.023053  min=-0.058289  max=+0.053374
  Near-zero (|x|<1e-4): weights 0.19%  biases 0.78%

--- fc2  Linear(256 -> 128) ---
  Weights: mean=+0.009993  std=0.094967  min=-0.481511  max=+0.567566  |w|_mean=0.073793
  Biases:  mean=+0.004879  std=0.065994  min=-0.140988  max=+0.180393
  Near-zero (|x|<1e-4): weights 0.10%  biases 0.00%

--- fc3  Linear(128 -> 10) ---
  Weights: mean=-0.049514  std=0.170451  min=-0.552233  max=+0.453748  |w|_mean=0.140224
  Biases:  mean=-0.004894  std=0.092467  min=-0.122941  max=+0.134560
  Near-zero (|x|<1e-4): weights 0.00%  biases 0.00%
  
'''







#---------------------
# --- Global weight distribution across all layers ---
all_weights = np.concatenate([li["weight"].ravel() for li in layer_info])
all_biases = np.concatenate([li["bias"].ravel() for li in layer_info])
print("\n" + "-" * 60)
print("GLOBAL (all layers combined)")
print(f"  All weights: mean={all_weights.mean():+.6f}  std={all_weights.std():.6f}")
print(f"  All biases:  mean={all_biases.mean():+.6f}  std={all_biases.std():.6f}")
print("-" * 60)


'''
------------------------------------------------------------
GLOBAL (all layers combined)
  All weights: mean=+0.002703  std=0.063357
  All biases:  mean=-0.002192  std=0.044735
------------------------------------------------------------
'''









#---------------------
# --- Activation snapshot on one test batch (how signals flow) ---
model.eval()
with torch.no_grad():
    sample_images, sample_labels = next(iter(test_loader))
    x = sample_images
    h1 = model.relu(model.fc1(x))
    h2 = model.relu(model.fc2(h1))
    logits = model.fc3(h2)

activations = {
    "input (flattened)": x.cpu().numpy(),
    "after fc1+ReLU": h1.cpu().numpy(),
    "after fc2+ReLU": h2.cpu().numpy(),
    "logits (fc3)": logits.cpu().numpy(),
}

print("\n" + "=" * 60)
print("ACTIVATION STATISTICS (one test batch, batch_size={})".format(sample_images.size(0)))
print("=" * 60)
for act_name, arr in activations.items():
    print(f"  {act_name:<22} shape={str(arr.shape):<20} "
          f"mean={arr.mean():+.4f}  std={arr.std():.4f}  "
          f"min={arr.min():+.4f}  max={arr.max():+.4f}  "
          f"dead_units(ReLU=0)={(arr == 0).mean()*100:.1f}%")


'''
============================================================
ACTIVATION STATISTICS (one test batch, batch_size=16)
============================================================
  input (flattened)      shape=(16, 784)            mean=-0.0392  std=0.9521  min=-0.4242  max=+2.8215  dead_units(ReLU=0)=0.0%
  after fc1+ReLU         shape=(16, 256)            mean=+0.9675  std=1.8789  min=+0.0000  max=+17.4630  dead_units(ReLU=0)=64.0%
  after fc2+ReLU         shape=(16, 128)            mean=+3.2048  std=3.8938  min=+0.0000  max=+19.0233  dead_units(ReLU=0)=38.5%
  logits (fc3)           shape=(16, 10)             mean=-20.3983  std=23.0291  min=-82.5738  max=+44.4388  dead_units(ReLU=0)=0.0%
'''







# ============================================
# Visualizations
# ============================================
n_layers = len(layer_info)
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

plt.suptitle("MNIST ANN — per-layer weight & bias histograms", fontsize=13, y=1.01)
plt.tight_layout()
if SAVE_FIG:
    plt.savefig("eda_layer_distributions.png", dpi=150, bbox_inches="tight")
print("\nSaved: eda_layer_distributions.png")
plt.show()








#---------------------
# --- Bar chart: mean |weight| and |bias| per layer ---
fig2, ax2 = plt.subplots(1, 2, figsize=(10, 4))
names = [li["name"] for li in layer_info]
mean_abs_w = [np.abs(li["weight"]).mean() for li in layer_info]
mean_abs_b = [np.abs(li["bias"]).mean() for li in layer_info]
std_w = [li["weight"].std() for li in layer_info]

ax2[0].bar(names, mean_abs_w, color="steelblue", edgecolor="black")
ax2[0].set_title("Mean |weight| per layer")
ax2[0].set_ylabel("mean |w|")

ax2[1].bar(names, mean_abs_b, color="darkorange", edgecolor="black")
ax2[1].set_title("Mean |bias| per layer")
ax2[1].set_ylabel("mean |b|")

plt.suptitle("MNIST ANN — average magnitude per layer", fontsize=12)
plt.tight_layout()
if SAVE_FIG:
    plt.savefig("eda_layer_magnitudes.png", dpi=150, bbox_inches="tight")
print("Saved: eda_layer_magnitudes.png")
plt.show()





#---------------------
# --- Weight std per layer (spread of learning) ---
fig3, ax3 = plt.subplots(figsize=(7, 4))
ax3.bar(names, std_w, color="seagreen", edgecolor="black")
ax3.set_title("Weight standard deviation per layer")
ax3.set_ylabel("std(weights)")
plt.tight_layout()
if SAVE_FIG:   
    plt.savefig("eda_weight_std_per_layer.png", dpi=150, bbox_inches="tight")
print("Saved: eda_weight_std_per_layer.png")
plt.show()






#---------------------
# --- Heatmaps: fc3 (10 classes × 128) is small enough; fc1 subset (first 32 neurons) ---
fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))

# fc3 full weight matrix (readable)
w3 = layer_info[-1]["weight"]  # fc3: (10, 128)
im0 = axes4[0].imshow(w3, aspect="auto", cmap="RdBu_r",
                       vmin=-np.percentile(np.abs(w3), 99),
                       vmax=np.percentile(np.abs(w3), 99))
axes4[0].set_title("fc3 weight matrix (10 outputs × 128 inputs)")
axes4[0].set_xlabel("input neuron (hidden2)")
axes4[0].set_ylabel("output class (0–9)")
plt.colorbar(im0, ax=axes4[0], fraction=0.046)

# fc1: first 32 output neurons × all 784 inputs (subset for visibility)
w1 = layer_info[0]["weight"][:32, :]  # (32, 784)
im1 = axes4[1].imshow(w1, aspect="auto", cmap="RdBu_r",
                       vmin=-np.percentile(np.abs(w1), 99),
                       vmax=np.percentile(np.abs(w1), 99))
axes4[1].set_title("fc1 weights — first 32 neurons × 784 pixels")
axes4[1].set_xlabel("flattened pixel index")
axes4[1].set_ylabel("hidden neuron (first 32)")
plt.colorbar(im1, ax=axes4[1], fraction=0.046)

plt.suptitle("MNIST ANN — weight heatmaps", fontsize=12)
plt.tight_layout()
if SAVE_FIG:
    plt.savefig("eda_weight_heatmaps.png", dpi=150, bbox_inches="tight")
print("Saved: eda_weight_heatmaps.png")
plt.show()




#---------------------
# --- Activation distributions on sample batch ---
fig5, axes5 = plt.subplots(2, 2, figsize=(10, 8))
for ax, (act_name, arr) in zip(axes5.ravel(), activations.items()):
    flat = arr.ravel()
    ax.hist(flat, bins=80, color="purple", alpha=0.75, edgecolor="white")
    ax.set_title(f"{act_name}\n(mean={flat.mean():.3f}, std={flat.std():.3f})")
    ax.set_xlabel("activation value")
    ax.set_ylabel("count")
plt.suptitle("Activation distributions (one test batch)", fontsize=12)
plt.tight_layout()
if SAVE_FIG:
    plt.savefig("eda_activation_distributions.png", dpi=150, bbox_inches="tight")
print("Saved: eda_activation_distributions.png")
plt.show()






#---------------------
# --- Optional: visualize fc1 weights as 28×28 filters (first 16 neurons) ---
fig6, axes6 = plt.subplots(4, 4, figsize=(8, 8))
w1_full = layer_info[0]["weight"]
for i, ax in enumerate(axes6.ravel()):
    filt = w1_full[i].reshape(28, 28)
    ax.imshow(filt, cmap="RdBu_r")
    ax.set_title(f"fc1 neuron {i}", fontsize=8)
    ax.axis("off")
plt.suptitle("fc1 — first 16 learned input filters (28×28)", fontsize=12)
plt.tight_layout()
if SAVE_FIG:
    plt.savefig("eda_fc1_filters.png", dpi=150, bbox_inches="tight")
print("Saved: eda_fc1_filters.png")
plt.show()

print("\nEDA complete. Figures saved in the project folder.")






# ============================================
# 8) Save Model
# ============================================

file_name = PTH_DIR / f"mnist_ann_{date_name}.pth"

torch.save(model.state_dict(), file_name)
print(f"Model saved as {file_name}")





# ============================================
# 9) Example Inference
# ============================================
example_img, _ = test_dataset[0]
output = model(example_img)
pred = output.argmax().item()

print("Example prediction digit:", pred)







