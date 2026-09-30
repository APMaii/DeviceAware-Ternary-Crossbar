from __future__ import annotations

'''
In The Name of God

Ali Pilehvar Meibody

Last Update : 07 sep 2026


06_Ideal_Cross_Sim.py




Now here after 04 which we Load out ANN, TNN and we make sure , we can again
use their loader and check Ideal CrossSim which is simialr to digital or not. 



Here with CrossSim we just create digital ANN (bitsliced) and Analog TNN (balanced)
only to see that everything is perfect and because it is ideal we must not see any
things difference to confirm that Cross Sim is sync with torch




Also from 05_Device_Modelign we extract Ron and Roff which also we know
they donot have any efffetc



Notes for later:
    Check this import from 04_....
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

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
from pathlib import Path

from paths import DATA_DIR, PROJECT_DIR, choose_checkpoint, ensure_cross_sim_on_path
ensure_cross_sim_on_path()

from simulator import CrossSimParameters
from simulator.parameters.xbar_parameters import ADCRangeLimits
from simulator.algorithms.dnn.torch.convert import from_torch

from simulator.algorithms.dnn.torch.convert import convertible_modules



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
SAVE_CHECKPOINTS = False



FIGURE_DPI = 150
THRESHOLD = 0.05
SMOOTH_TW_WIDTH = 0.7
DEVICE = torch.device("cpu")





# Listed from Pth_Models. Enter keeps the newest file.
ANN_CHECKPOINT = choose_checkpoint("ann")
TNN_CHECKPOINT = choose_checkpoint("tnn")



R_MIN=1.410360e+09  
R_MAX=4.429246e+09 



#=========================================================
'''                    ANN TNN LOAD                '''
#=========================================================

from pathlib import Path
import sys

import importlib
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
_load = importlib.import_module("04_Load_ANN_TNN")
load_ann = _load.load_ann
load_tnn_ternary = _load.load_tnn_ternary
print_model_summary = _load.print_model_summary









net_ann, ann_path, ann_meta = load_ann(ANN_CHECKPOINT)
print_model_summary(net_ann, "ANN (full-precision)", ann_path, ann_meta)

'''
ANN (full-precision)
  checkpoint: BASE_ANN_mnist_lr0.0001_ep100_seed42_20260522.pth
  parameters: 235,146
  saved test accuracy: 98.00%
  layers: fc1 (256, 784), fc2 (128, 256), fc3 (10, 128)
  
'''

net_tnn, tnn_path, tnn_meta = load_tnn_ternary(TNN_CHECKPOINT)
print_model_summary(net_tnn, "TNN (ternary -1/0/+1)", tnn_path, tnn_meta)
'''
TNN (ternary -1/0/+1)
  checkpoint: TERNARY_ONLY_mnist_tw0.7_th0.05_seed42_20260522.pth
  parameters: 235,146
  saved test accuracy: 93.78%
  layers: fc1 (256, 784), fc2 (128, 256), fc3 (10, 128)
  
'''




#=========================================================
'''                  Load Data               '''
#=========================================================


#Load MNIST Test data (same preprocessing as Base_ANN_TNN / ANN_MNIST)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.view(-1))   # flatten 28x28 to 784
])

test_dataset = datasets.MNIST(
    root=str(DATA_DIR),
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False
)



#ACcuracy fucntion
def evaluate(model, test_loader, device="cpu"):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            pred = output.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    return 100 * correct / total






# ============================================
# ============================================
'''        IDEAL Digital ANN               '''
# ============================================
# ============================================



cs_params_sliced = CrossSimParameters()

# Array size
cs_params_sliced.core.rows_max = 1024
cs_params_sliced.core.cols_max = 1024
cs_params_sliced.core.weight_bits =8


# important slicing settings
cs_params_sliced.core.style = 2 #Bitsliced
cs_params_sliced.core.bit_sliced.style = 1 #Bitsliced for each --> BALANCED
cs_params_sliced.core.bit_sliced.num_slices = 8

# Device resistance range
# Temporary values first; later replace with your measured LRS/HRS

#---------**** CHANGE IF YOU NEEED---------------------------
cs_params_sliced.xbar.device.Rmin = R_MIN  # LRS / ON
cs_params_sliced.xbar.device.Rmax = R_MAX  # HRS / OFF
cs_params_sliced.xbar.device.cell_bits = 1
cs_params_sliced.xbar.device.Vread = 1.0



# Start ideal: no programming error, read noise, drift
cs_params_sliced.xbar.device.programming_error.enable = False
cs_params_sliced.xbar.device.read_noise.enable = False
cs_params_sliced.xbar.device.drift_error.enable = False

# Start ideal ADC/DAC
cs_params_sliced.xbar.adc.mvm.bits = 0
cs_params_sliced.xbar.adc.vmm.bits = 0
cs_params_sliced.xbar.dac.mvm.bits = 0
cs_params_sliced.xbar.dac.vmm.bits = 0



cs_params_sliced.xbar.array.parasitics.enable = False

# example small parasitic resistance
cs_params_sliced.xbar.array.parasitics.Rp_row = 0
cs_params_sliced.xbar.array.parasitics.Rp_col = 0

# optional terminal parasitics, keep zero first
cs_params_sliced.xbar.array.parasitics.Rp_row_terminal = 0
cs_params_sliced.xbar.array.parasitics.Rp_col_terminal = 0

# keep default
cs_params_sliced.xbar.array.parasitics.current_from_input = True
cs_params_sliced.xbar.array.parasitics.selected_rows = "top"

#print(cs_params_sliced)

#check convertable layers
convertible_modules(net_ann)

#convert pytorch to crossim analog model
analog_ideal_ann_net = from_torch(net_ann, cs_params_sliced)
analog_ideal_ann_net.eval()

#Compare digital vs CrossSim analog
digital_ideal_ann_acc = evaluate(net_ann, test_loader, device="cpu")
analog_ideal_ann_acc = evaluate(analog_ideal_ann_net, test_loader, device="cpu")

print(f"Digital accuracy: {digital_ideal_ann_acc:.2f}%")
print(f"CrossSim analog accuracy: {analog_ideal_ann_acc:.2f}%")

'''
Digital accuracy: 98.00%
CrossSim analog accuracy: 98.01%


24 may
Digital accuracy: 98.00%
CrossSim analog accuracy: 98.01%

'''



# ============================================
# ============================================
'''          IDEAL Digital TNN                '''
# ============================================
# ============================================

cs_params_ternary = CrossSimParameters()

# Array size
cs_params_ternary.core.rows_max = 1024
cs_params_ternary.core.cols_max = 1024


# important slicing settings
cs_params_ternary.core.style = 1  # CoreStyle.BALANCED


#here we must say why offset is not good, why only one-side not two side (in term of energy and ...)
# Balanced settings
cs_params_ternary.core.balanced.style = 1
cs_params_ternary.core.balanced.interleaved_posneg = False
cs_params_ternary.core.balanced.subtract_current_in_xbar = True


# No bit slicing
cs_params_ternary.core.weight_bits = 0
cs_params_ternary.core.bit_sliced.num_slices = 1


# Device resistance range
# Temporary values first; later replace with your measured LRS/HRS

#---------**** CHANGE IF YOU NEEED---------------------------
cs_params_ternary.xbar.device.Rmin = R_MIN   # LRS / ON
cs_params_ternary.xbar.device.Rmax = R_MAX  # HRS / OFF
cs_params_ternary.xbar.device.cell_bits = 1
cs_params_ternary.xbar.device.Vread = 1.0



# Start ideal: no programming error, read noise, drift
cs_params_ternary.xbar.device.programming_error.enable = False
cs_params_ternary.xbar.device.read_noise.enable = False
cs_params_ternary.xbar.device.drift_error.enable = False

# Start ideal ADC/DAC
cs_params_ternary.xbar.adc.mvm.bits = 0
cs_params_ternary.xbar.adc.vmm.bits = 0
cs_params_ternary.xbar.dac.mvm.bits = 0
cs_params_ternary.xbar.dac.vmm.bits = 0



cs_params_ternary.xbar.array.parasitics.enable = False

# example small parasitic resistance
cs_params_ternary.xbar.array.parasitics.Rp_row = 0
cs_params_ternary.xbar.array.parasitics.Rp_col = 0

# optional terminal parasitics, keep zero first
cs_params_ternary.xbar.array.parasitics.Rp_row_terminal = 0
cs_params_ternary.xbar.array.parasitics.Rp_col_terminal = 0

# keep default
cs_params_ternary.xbar.array.parasitics.current_from_input = True
cs_params_ternary.xbar.array.parasitics.selected_rows = "top"

#check convertable layers
convertible_modules(net_tnn)

#convert pytorch to crossim analog model
analog_ideal_tnn_net = from_torch(net_tnn, cs_params_ternary)
analog_ideal_tnn_net.eval()



#Compare digital vs CrossSim analog
digital_ideal_tnn_acc = evaluate(net_tnn, test_loader, device="cpu")
analog_ideal_tnn_acc = evaluate(analog_ideal_tnn_net, test_loader, device="cpu")

print(f"Digital accuracy: {digital_ideal_tnn_acc:.2f}%")
print(f"CrossSim analog accuracy: {analog_ideal_tnn_acc:.2f}%")

'''
Digital accuracy: 93.78%
CrossSim analog accuracy: 93.78%


24 may
Digital accuracy: 93.78%
CrossSim analog accuracy: 93.78%

'''









# ============================================
'''          Visualziation              '''

# ============================================
_ACADEMIC_RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "normal",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "legend.fontsize": 10,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#CCCCCC",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}
_COLOR_DIGITAL = "#2C5282"
_COLOR_ANALOG = "#C05621"
_COLOR_SCATTER = "#2C5282"

acc_data = {
    "ANN": {
        "Digital": digital_ideal_ann_acc,
        "Ideal CrossSim": analog_ideal_ann_acc
    },
    "TNN": {
        "Digital": digital_ideal_tnn_acc,
        "Ideal CrossSim": analog_ideal_tnn_acc
    }
}







#=================================================
'''     Figure  Accuacy comparison        '''
#=================================================

models = list(acc_data.keys())
digital_acc = [acc_data[m]["Digital"] for m in models]
analog_acc = [acc_data[m]["Ideal CrossSim"] for m in models]

x = np.arange(len(models))
width = 0.35

with plt.rc_context(_ACADEMIC_RC):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        x - width / 2, digital_acc, width,
        label="Digital PyTorch", color=_COLOR_DIGITAL,
        edgecolor="white", linewidth=0.8, zorder=3,
    )
    ax.bar(
        x + width / 2, analog_acc, width,
        label="Ideal CrossSim", color=_COLOR_ANALOG,
        edgecolor="white", linewidth=0.8, zorder=3,
    )

    ax.set_xticks(x, models)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Digital vs Ideal CrossSim Accuracy")
    ax.set_ylim(90, 100)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.45, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, v in enumerate(digital_acc):
        ax.text(
            i - width / 2, v + 0.05, f"{v:.2f}%",
            ha="center", fontsize=9, color=_COLOR_DIGITAL,
        )

    for i, v in enumerate(analog_acc):
        ax.text(
            i + width / 2, v + 0.05, f"{v:.2f}%",
            ha="center", fontsize=9, color=_COLOR_ANALOG,
        )

    fig.tight_layout()
    plt.show()





#=================================================
'''     Figure  Confusion Matrix         '''
#=================================================
#----------------------------------------------------------------------
#Figure 2 — Confusion Matrix: Digital vs Ideal CrossSim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch

def get_predictions(model, loader, device="cpu"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            pred = out.argmax(dim=1)

            all_preds.append(pred.cpu())
            all_labels.append(y.cpu())

    return torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy()


# ANN
y_true_ann, y_pred_digital_ann = get_predictions(net_ann, test_loader)
_, y_pred_analog_ann = get_predictions(analog_ideal_ann_net, test_loader)

cm_digital_ann = confusion_matrix(y_true_ann, y_pred_digital_ann)
cm_analog_ann = confusion_matrix(y_true_ann, y_pred_analog_ann)

ConfusionMatrixDisplay(cm_digital_ann).plot()
plt.title("ANN Digital Confusion Matrix")
plt.show()

ConfusionMatrixDisplay(cm_analog_ann).plot()
plt.title("ANN Ideal CrossSim Confusion Matrix")
plt.show()


# TNN
y_true_tnn, y_pred_digital_tnn = get_predictions(net_tnn, test_loader)
_, y_pred_analog_tnn = get_predictions(analog_ideal_tnn_net, test_loader)

cm_digital_tnn = confusion_matrix(y_true_tnn, y_pred_digital_tnn)
cm_analog_tnn = confusion_matrix(y_true_tnn, y_pred_analog_tnn)

ConfusionMatrixDisplay(cm_digital_tnn).plot()
plt.title("TNN Digital Confusion Matrix")
plt.show()

ConfusionMatrixDisplay(cm_analog_tnn).plot()
plt.title("TNN Ideal CrossSim Confusion Matrix")
plt.show()














#=================================================
'''     Figure  Scatter plot ann-tnn        '''
#=================================================

#--------------------------------------
import numpy as np
import matplotlib.pyplot as plt
# (uses _ACADEMIC_RC / _COLOR_* from accuracy plot block above)
# =========================
# Collect digital layers
# =========================

def collect_digital_layers(model):
    layers = []

    for name, module in model.named_modules():
        if hasattr(module, "weight") and module.weight is not None:
            w = module.weight.detach().cpu().numpy()

            if w.ndim >= 2:
                layers.append({
                    "name": name,
                    "weight": np.squeeze(w)
                })

    return layers



# Collect analog layers
def collect_analog_layers(model):
    layers = []

    for name, module in model.named_modules():
        analog_w = None

        if hasattr(module, "core"):
            try:
                analog_w = module.core.get_matrix()
            except Exception:
                pass

        if analog_w is None and hasattr(module, "analog_core"):
            try:
                analog_w = module.analog_core.get_matrix()
            except Exception:
                pass

        if analog_w is not None:
            analog_w = np.squeeze(np.array(analog_w))

            if analog_w.ndim >= 2:
                layers.append({
                    "name": name,
                    "weight": analog_w
                })

    return layers


digital_ann_layers = collect_digital_layers(net_ann)
digital_tnn_layers = collect_digital_layers(net_tnn)

analog_ann_layers = collect_analog_layers(analog_ideal_ann_net)
analog_tnn_layers = collect_analog_layers(analog_ideal_tnn_net)


print("ANN digital layers:")
for i, layer in enumerate(digital_ann_layers):
    print(i, layer["name"], layer["weight"].shape)

print("\nANN analog layers:")
for i, layer in enumerate(analog_ann_layers):
    print(i, layer["name"], layer["weight"].shape)

print("\nTNN digital layers:")
for i, layer in enumerate(digital_tnn_layers):
    print(i, layer["name"], layer["weight"].shape)

print("\nTNN analog layers:")
for i, layer in enumerate(analog_tnn_layers):
    print(i, layer["name"], layer["weight"].shape)
    
    

def plot_weight_scatter_all_layers(digital_layers, analog_layers, model_name, num_layers=3):
    n = min(num_layers, len(digital_layers), len(analog_layers))

    for i in range(n):
        digital_w = np.squeeze(digital_layers[i]["weight"]).flatten()
        analog_w = np.squeeze(analog_layers[i]["weight"]).flatten()

        min_len = min(len(digital_w), len(analog_w))
        digital_w = digital_w[:min_len]
        analog_w = analog_w[:min_len]

        with plt.rc_context(_ACADEMIC_RC):
            fig, ax = plt.subplots(figsize=(5.5, 5.5))
            ax.scatter(
                digital_w, analog_w,
                s=4, alpha=0.35, color=_COLOR_SCATTER,
                edgecolors="none", rasterized=True,
            )
            ax.set_xlabel("Digital Weight")
            ax.set_ylabel("Programmed Analog Weight")
            ax.set_title(f"{model_name} Layer {i+1}: Digital vs Analog Weights")
            ax.set_aspect("equal", adjustable="box")
            ax.set_axisbelow(True)
            ax.grid(alpha=0.45, linestyle="--", linewidth=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            plt.show()

        error = analog_w - digital_w

        print(f"{model_name} Layer {i+1}")
        print("Digital layer:", digital_layers[i]["name"])
        print("Analog layer:", analog_layers[i]["name"])
        print("Correlation:", np.corrcoef(digital_w, analog_w)[0, 1])
        print("Mean absolute error:", np.mean(np.abs(error)))
        print("Max absolute error:", np.max(np.abs(error)))
        print("-" * 60)


plot_weight_scatter_all_layers(
    digital_ann_layers,
    analog_ann_layers,
    model_name="ANN",
    num_layers=3
)

plot_weight_scatter_all_layers(
    digital_tnn_layers,
    analog_tnn_layers,
    model_name="TNN",
    num_layers=3
)






















#=================================================
'''     Figure  ANN,TNN Layer heatmap         '''
#=================================================
#-----------------------------------------------------------
# 1. Collect digital weights

def collect_digital_weights(model):
    layers = []

    for name, module in model.named_modules():
        if hasattr(module, "weight") and module.weight is not None:
            w = module.weight.detach().cpu().numpy()
            if w.ndim >= 2:
                layers.append({
                    "name": name,
                    "weight": np.squeeze(w)
                })

    return layers


# 2. Collect analog weights
def collect_analog_weights(model):
    layers = []

    for name, module in model.named_modules():
        analog_matrix = None

        if hasattr(module, "core"):
            try:
                analog_matrix = module.core.get_matrix()
            except Exception:
                pass

        elif hasattr(module, "analog_core"):
            try:
                analog_matrix = module.analog_core.get_matrix()
            except Exception:
                pass

        if analog_matrix is not None:
            analog_matrix = np.squeeze(np.array(analog_matrix))

            if analog_matrix.ndim >= 2:
                layers.append({
                    "name": name,
                    "weight": analog_matrix
                })

    return layers


digital_ann_layers = collect_digital_weights(net_ann)
digital_tnn_layers = collect_digital_weights(net_tnn)

analog_ann_layers = collect_analog_weights(analog_ideal_ann_net)
analog_tnn_layers = collect_analog_weights(analog_ideal_tnn_net)


print("ANN digital layers:")
for i, layer in enumerate(digital_ann_layers):
    print(i, layer["name"], layer["weight"].shape)

print("\nANN analog layers:")
for i, layer in enumerate(analog_ann_layers):
    print(i, layer["name"], layer["weight"].shape)

print("\nTNN digital layers:")
for i, layer in enumerate(digital_tnn_layers):
    print(i, layer["name"], layer["weight"].shape)

print("\nTNN analog layers:")
for i, layer in enumerate(analog_tnn_layers):
    print(i, layer["name"], layer["weight"].shape)
    
    
# 3. Heatmap function
def plot_layer_heatmap_pair(digital_w, analog_w, model_name, layer_idx):
    digital_w = np.squeeze(digital_w)
    analog_w = np.squeeze(analog_w)

    min_rows = min(digital_w.shape[0], analog_w.shape[0])
    min_cols = min(digital_w.shape[1], analog_w.shape[1])

    digital_w = digital_w[:min_rows, :min_cols]
    analog_w = analog_w[:min_rows, :min_cols]

    error = analog_w - digital_w

    # Same scale for all three heatmaps
    vmax = max(abs(digital_w).max(), abs(analog_w).max())

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    im0 = axes[0].imshow(digital_w, aspect="auto", vmin=-vmax, vmax=vmax)
    axes[0].set_title(f"{model_name} Layer {layer_idx}: Digital Weights")
    axes[0].set_xlabel("Input Neurons")
    axes[0].set_ylabel("Output Neurons")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(analog_w, aspect="auto", vmin=-vmax, vmax=vmax)
    axes[1].set_title(f"{model_name} Layer {layer_idx}: Analog Weights")
    axes[1].set_xlabel("Input Neurons")
    axes[1].set_ylabel("Output Neurons")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(error, aspect="auto", vmin=-vmax, vmax=vmax)
    axes[2].set_title(f"{model_name} Layer {layer_idx}: Analog - Digital")
    axes[2].set_xlabel("Input Neurons")
    axes[2].set_ylabel("Output Neurons")
    plt.colorbar(im2, ax=axes[2])

    plt.suptitle(f"{model_name} Layer {layer_idx} Weight Mapping Validation", fontsize=14)
    plt.tight_layout()
    plt.show()

    print(f"{model_name} Layer {layer_idx}")
    print("Mean absolute error:", np.mean(np.abs(error)))
    print("Max absolute error:", np.max(np.abs(error)))
    print("Digital shape:", digital_w.shape)
    print("Analog shape:", analog_w.shape)
    print("-" * 60)
    
    
# 4. Generate 6 heatmap figures

num_layers = 3

for i in range(num_layers):
    plot_layer_heatmap_pair(
        digital_ann_layers[i]["weight"],
        analog_ann_layers[i]["weight"],
        model_name="ANN",
        layer_idx=i + 1
    )

for i in range(num_layers):
    plot_layer_heatmap_pair(
        digital_tnn_layers[i]["weight"],
        analog_tnn_layers[i]["weight"],
        model_name="TNN",
        layer_idx=i + 1
    )



# 3. Heatmap function
def safe_plot_layer_heatmap_pair(digital_w, analog_w, model_name, layer_idx):
    digital_w = np.squeeze(digital_w)
    analog_w = np.squeeze(analog_w)

    min_rows = min(digital_w.shape[0], analog_w.shape[0])
    min_cols = min(digital_w.shape[1], analog_w.shape[1])

    digital_w = digital_w[:min_rows, :min_cols]
    analog_w = analog_w[:min_rows, :min_cols]

    error = analog_w - digital_w

    # Same scale for all three heatmaps
    vmax = max(abs(digital_w).max(), abs(analog_w).max())

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    im0 = axes[0].imshow(
        digital_w, aspect="auto",
        vmin=-vmax, vmax=vmax,
        cmap="cividis"
    )
    axes[0].set_title(f"{model_name} Layer {layer_idx}: Digital Weights")
    axes[0].set_xlabel("Input Neurons")
    axes[0].set_ylabel("Output Neurons")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        analog_w, aspect="auto",
        vmin=-vmax, vmax=vmax,
        cmap="cividis"
    )
    axes[1].set_title(f"{model_name} Layer {layer_idx}: Analog Weights")
    axes[1].set_xlabel("Input Neurons")
    axes[1].set_ylabel("Output Neurons")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(
        error, aspect="auto",
        vmin=-vmax, vmax=vmax,
        cmap="cividis"
    )
    axes[2].set_title(f"{model_name} Layer {layer_idx}: Analog - Digital")
    axes[2].set_xlabel("Input Neurons")
    axes[2].set_ylabel("Output Neurons")
    plt.colorbar(im2, ax=axes[2])

    plt.suptitle(f"{model_name} Layer {layer_idx} Weight Mapping Validation", fontsize=14)
    plt.tight_layout()
    plt.show()

    print(f"{model_name} Layer {layer_idx}")
    print("Mean absolute error:", np.mean(np.abs(error)))
    print("Max absolute error:", np.max(np.abs(error)))
    print("Digital shape:", digital_w.shape)
    print("Analog shape:", analog_w.shape)
    print("-" * 60)


num_layers = 3

for i in range(num_layers):
    safe_plot_layer_heatmap_pair(
        digital_ann_layers[i]["weight"],
        analog_ann_layers[i]["weight"],
        model_name="ANN",
        layer_idx=i + 1
    )

for i in range(num_layers):
    safe_plot_layer_heatmap_pair(
        digital_tnn_layers[i]["weight"],
        analog_tnn_layers[i]["weight"],
        model_name="TNN",
        layer_idx=i + 1
    )






#=================================================
'''     Figure  ANN Layer heatmap         '''
#=================================================
#------------------------------------------------------------------------------
# 5. ANN bit-sliced weight planes (CrossSim BitslicedCore)
# Find bit-sliced AnalogCore layers

def collect_bitsliced_analog_layers(model):
    layers = []

    for name, module in model.named_modules():
        core_obj = None

        if hasattr(module, "core"):
            core_obj = module.core

        elif hasattr(module, "analog_core"):
            core_obj = module.analog_core

        if core_obj is None:
            continue

        try:
            wrapper_core = core_obj.cores[0][0]

            if hasattr(wrapper_core, "core_slices"):
                layers.append({
                    "name": name,
                    "analog_core": core_obj,
                    "wrapper_core": wrapper_core
                })

        except Exception:
            pass

    return layers


bitsliced_ann_layers = collect_bitsliced_analog_layers(analog_ideal_ann_net)

print("Bit-sliced ANN layers found:")
for i, layer in enumerate(bitsliced_ann_layers):
    wc = layer["wrapper_core"]
    print(
        i,
        layer["name"],
        "Nslices =", wc.Nslices,
        "W_shape =", wc.W_shape
    )
    


def plot_bitslices_for_layer(layer_info, layer_idx, normalize_by_Rmin=True):
    wc = layer_info["wrapper_core"]
    name = layer_info["name"]

    Nslices = wc.Nslices

    if normalize_by_Rmin:
        scale = wc.params.xbar.device.Rmin
        scale_label = "Normalized conductance / Rmin"
    else:
        scale = 1.0
        scale_label = "Stored slice matrix"

    for s in range(Nslices):
        G_pos = np.array(wc.core_slices[s][0].matrix) / scale
        G_neg = np.array(wc.core_slices[s][1].matrix) / scale

        G_combined = G_pos - G_neg
        G_concat = np.concatenate((G_pos, G_neg), axis=1)

        vmax = max(abs(G_pos).max(), abs(G_neg).max(), abs(G_combined).max())

        fig, axes = plt.subplots(1, 4, figsize=(20, 4))

        im0 = axes[0].imshow(G_pos, aspect="auto", vmin=-vmax, vmax=vmax)
        axes[0].set_title(f"Slice {s}: Positive")
        axes[0].set_xlabel("Input")
        axes[0].set_ylabel("Output")
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(G_neg, aspect="auto", vmin=-vmax, vmax=vmax)
        axes[1].set_title(f"Slice {s}: Negative")
        axes[1].set_xlabel("Input")
        axes[1].set_ylabel("Output")
        plt.colorbar(im1, ax=axes[1])

        im2 = axes[2].imshow(G_combined, aspect="auto", vmin=-vmax, vmax=vmax)
        axes[2].set_title(f"Slice {s}: Positive - Negative")
        axes[2].set_xlabel("Input")
        axes[2].set_ylabel("Output")
        plt.colorbar(im2, ax=axes[2])

        im3 = axes[3].imshow(G_concat, aspect="auto")
        axes[3].set_title(f"Slice {s}: Positive | Negative")
        axes[3].set_xlabel("Input")
        axes[3].set_ylabel("Output")
        plt.colorbar(im3, ax=axes[3])

        plt.suptitle(
            f"ANN Layer {layer_idx} Bit Slice {s} Conductance Mapping\n{name}",
            fontsize=14
        )
        plt.tight_layout()
        plt.show()

        print(f"ANN Layer {layer_idx}, Slice {s}")
        print("Positive shape:", G_pos.shape)
        print("Negative shape:", G_neg.shape)
        print("Combined shape:", G_combined.shape)
        print("Positive min/max:", G_pos.min(), G_pos.max())
        print("Negative min/max:", G_neg.min(), G_neg.max())
        print("Combined min/max:", G_combined.min(), G_combined.max())
        print("-" * 60)
        

num_layers = 3

for layer_idx in range(num_layers):
    plot_bitslices_for_layer(
        bitsliced_ann_layers[layer_idx],
        layer_idx=layer_idx + 1,
        normalize_by_Rmin=True
    )
    
    
    
    
    
    
    
def safe_plot_bitslices_for_layer(layer_info, layer_idx, normalize_by_Rmin=True):
    wc = layer_info["wrapper_core"]
    name = layer_info["name"]

    Nslices = wc.Nslices

    if normalize_by_Rmin:
        scale = wc.params.xbar.device.Rmin
        scale_label = "Normalized conductance / Rmin"
    else:
        scale = 1.0
        scale_label = "Stored slice matrix"

    for s in range(Nslices):
        G_pos = np.array(wc.core_slices[s][0].matrix) / scale
        G_neg = np.array(wc.core_slices[s][1].matrix) / scale

        G_combined = G_pos - G_neg
        G_concat = np.concatenate((G_pos, G_neg), axis=1)

        vmax = max(abs(G_pos).max(), abs(G_neg).max(), abs(G_combined).max())

        fig, axes = plt.subplots(1, 4, figsize=(20, 4))

        im0 = axes[0].imshow(G_pos, aspect="auto", vmin=-vmax, vmax=vmax, cmap="cividis")
        axes[0].set_title(f"Slice {s}: Positive")
        axes[0].set_xlabel("Input")
        axes[0].set_ylabel("Output")
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(G_neg, aspect="auto", vmin=-vmax, vmax=vmax, cmap="cividis")
        axes[1].set_title(f"Slice {s}: Negative")
        axes[1].set_xlabel("Input")
        axes[1].set_ylabel("Output")
        plt.colorbar(im1, ax=axes[1])

        im2 = axes[2].imshow(G_combined, aspect="auto", vmin=-vmax, vmax=vmax, cmap="cividis")
        axes[2].set_title(f"Slice {s}: Positive - Negative")
        axes[2].set_xlabel("Input")
        axes[2].set_ylabel("Output")
        plt.colorbar(im2, ax=axes[2])

        im3 = axes[3].imshow(G_concat, aspect="auto", cmap="cividis")
        axes[3].set_title(f"Slice {s}: Positive | Negative")
        axes[3].set_xlabel("Input")
        axes[3].set_ylabel("Output")
        plt.colorbar(im3, ax=axes[3])

        plt.suptitle(
            f"ANN Layer {layer_idx} Bit Slice {s} Conductance Mapping\n{name}",
            fontsize=14
        )
        plt.tight_layout()
        plt.show()

        print(f"ANN Layer {layer_idx}, Slice {s}")
        print("Positive shape:", G_pos.shape)
        print("Negative shape:", G_neg.shape)
        print("Combined shape:", G_combined.shape)
        print("Positive min/max:", G_pos.min(), G_pos.max())
        print("Negative min/max:", G_neg.min(), G_neg.max())
        print("Combined min/max:", G_combined.min(), G_combined.max())
        print("-" * 60)
    
    
    
    
    
    
    

num_layers = 3

for layer_idx in range(num_layers):
    safe_plot_bitslices_for_layer(
        bitsliced_ann_layers[layer_idx],
        layer_idx=layer_idx + 1,
        normalize_by_Rmin=True
    )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
#=================================================
'''     Figure  TNN Layer heatmap         '''
#=================================================

#---------------------------------------------------
# 1. Collect balanced TNN layers

def collect_balanced_analog_layers(model):
    layers = []

    for name, module in model.named_modules():
        core_obj = None

        if hasattr(module, "core"):
            core_obj = module.core

        elif hasattr(module, "analog_core"):
            core_obj = module.analog_core

        if core_obj is None:
            continue

        try:
            wrapper_core = core_obj.cores[0][0]

            if hasattr(wrapper_core, "core_pos") and hasattr(wrapper_core, "core_neg"):
                layers.append({
                    "name": name,
                    "analog_core": core_obj,
                    "wrapper_core": wrapper_core
                })

        except Exception:
            pass

    return layers


balanced_tnn_layers = collect_balanced_analog_layers(analog_ideal_tnn_net)

print("Balanced TNN layers found:")
for i, layer in enumerate(balanced_tnn_layers):
    wc = layer["wrapper_core"]
    print(
        i,
        layer["name"],
        "W_shape =", wc.W_shape,
        "core_pos shape =", wc.core_pos.matrix.shape,
        "core_neg shape =", wc.core_neg.matrix.shape
    )
    
# 2. Plot TNN balanced conductance maps

def plot_balanced_tnn_layer(layer_info, layer_idx, normalize_by_Rmin=True):
    wc = layer_info["wrapper_core"]
    name = layer_info["name"]

    if normalize_by_Rmin:
        scale = wc.params.xbar.device.Rmin
        scale_label = "Conductance / Rmin"
    else:
        scale = 1.0
        scale_label = "Stored matrix"

    G_pos = np.array(wc.core_pos.matrix) / scale
    G_neg = np.array(wc.core_neg.matrix) / scale

    G_effective = G_pos - G_neg
    G_concat = np.concatenate((G_pos, G_neg), axis=1)

    vmax = max(
        abs(G_pos).max(),
        abs(G_neg).max(),
        abs(G_effective).max()
    )

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    im0 = axes[0].imshow(G_pos, aspect="auto", vmin=-vmax, vmax=vmax)
    axes[0].set_title("Positive Core")
    axes[0].set_xlabel("Input")
    axes[0].set_ylabel("Output")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(G_neg, aspect="auto", vmin=-vmax, vmax=vmax)
    axes[1].set_title("Negative Core")
    axes[1].set_xlabel("Input")
    axes[1].set_ylabel("Output")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(G_effective, aspect="auto", vmin=-vmax, vmax=vmax)
    axes[2].set_title("Effective: Positive - Negative")
    axes[2].set_xlabel("Input")
    axes[2].set_ylabel("Output")
    plt.colorbar(im2, ax=axes[2])

    im3 = axes[3].imshow(G_concat, aspect="auto")
    axes[3].set_title("Positive | Negative")
    axes[3].set_xlabel("Input")
    axes[3].set_ylabel("Output")
    plt.colorbar(im3, ax=axes[3])

    plt.suptitle(
        f"TNN Layer {layer_idx} Balanced Conductance Mapping\n{name}",
        fontsize=14
    )
    plt.tight_layout()
    plt.show()

    print(f"TNN Layer {layer_idx}")
    print("Layer name:", name)
    print("Scale:", scale_label)
    print("Positive shape:", G_pos.shape)
    print("Negative shape:", G_neg.shape)
    print("Effective shape:", G_effective.shape)
    print("Positive min/max:", G_pos.min(), G_pos.max())
    print("Negative min/max:", G_neg.min(), G_neg.max())
    print("Effective min/max:", G_effective.min(), G_effective.max())
    print("-" * 60)
    
    

# 3. Generate plots for all 3 TNN layers

num_layers = 3

for layer_idx in range(num_layers):
    plot_balanced_tnn_layer(
        balanced_tnn_layers[layer_idx],
        layer_idx=layer_idx + 1,
        normalize_by_Rmin=True
    )





def safe_plot_balanced_tnn_layer(layer_info, layer_idx, normalize_by_Rmin=True):
    wc = layer_info["wrapper_core"]
    name = layer_info["name"]

    if normalize_by_Rmin:
        scale = wc.params.xbar.device.Rmin
        scale_label = "Conductance / Rmin"
    else:
        scale = 1.0
        scale_label = "Stored matrix"

    G_pos = np.array(wc.core_pos.matrix) / scale
    G_neg = np.array(wc.core_neg.matrix) / scale

    G_effective = G_pos - G_neg
    G_concat = np.concatenate((G_pos, G_neg), axis=1)

    vmax = max(
        abs(G_pos).max(),
        abs(G_neg).max(),
        abs(G_effective).max()
    )

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    im0 = axes[0].imshow(
        G_pos, aspect="auto",
        vmin=-vmax, vmax=vmax,
        cmap="cividis"
    )
    axes[0].set_title("Positive Core")
    axes[0].set_xlabel("Input")
    axes[0].set_ylabel("Output")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        G_neg, aspect="auto",
        vmin=-vmax, vmax=vmax,
        cmap="cividis"
    )
    axes[1].set_title("Negative Core")
    axes[1].set_xlabel("Input")
    axes[1].set_ylabel("Output")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(
        G_effective, aspect="auto",
        vmin=-vmax, vmax=vmax,
        cmap="cividis"
    )
    axes[2].set_title("Effective: Positive - Negative")
    axes[2].set_xlabel("Input")
    axes[2].set_ylabel("Output")
    plt.colorbar(im2, ax=axes[2])

    im3 = axes[3].imshow(
        G_concat, aspect="auto",
        cmap="cividis"
    )
    axes[3].set_title("Positive | Negative")
    axes[3].set_xlabel("Input")
    axes[3].set_ylabel("Output")
    plt.colorbar(im3, ax=axes[3])

    plt.suptitle(
        f"TNN Layer {layer_idx} Balanced Conductance Mapping\n{name}",
        fontsize=14
    )
    plt.tight_layout()
    plt.show()

    print(f"TNN Layer {layer_idx}")
    print("Layer name:", name)
    print("Scale:", scale_label)
    print("Positive shape:", G_pos.shape)
    print("Negative shape:", G_neg.shape)
    print("Effective shape:", G_effective.shape)
    print("Positive min/max:", G_pos.min(), G_pos.max())
    print("Negative min/max:", G_neg.min(), G_neg.max())
    print("Effective min/max:", G_effective.min(), G_effective.max())
    print("-" * 60)


# 3. Generate plots for all 3 TNN layers

num_layers = 3

for layer_idx in range(num_layers):
    safe_plot_balanced_tnn_layer(
        balanced_tnn_layers[layer_idx],
        layer_idx=layer_idx + 1,
        normalize_by_Rmin=True
    )

