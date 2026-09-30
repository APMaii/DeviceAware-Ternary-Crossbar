from __future__ import annotations

'''
In The Name of God

Ali Pilehvar Meibody

Last Update : 07 sep 2026



07b_Real_Cross_Sim_TNN_vs_ANN.py


after modeling programmign error and drift in file 05 , here in file 06 we
got ideal ANN and TNN to only see that everything is work because
the digital and ideal must be same to prove cross sim is sync with pytorch
and even the Rmin Rmax and any devcie related doesn't matter becuase they
were Ideal.


Now in 07 we want to consider Rmin, Rmax and also Programming error
to consider that our programming error is ok or not.



First we have ideal (from 06) for ANN and TNN.
We implement SONOS and APM to compare with each others. we found APM is reasonable

then we compare the TNN and ANN in terms of configurations. and we decide
to go with TNN.
    
  
    
  
    
  
    
  
    
  
    
Next step (26may) consider always we check Rmin , Rmax
also we must cheange the figurs of here .


Inja bayad biaym real ro dar biarim kamel hame ax haro
badesh TNN comparison cost o ina k behtar bshe

exact effect of programming
exact effect of drift

ADC/DAC

Effect of paraistics only


final real case 


Device-Aware Training  


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
TNN_CHECKPOINT = choose_checkpoint("tnn")



R_MIN=1.410360e+09  
R_MAX=4.429246e+09 


#=========================================================
'''                    TNN LOAD                    '''
#=========================================================

from pathlib import Path
import sys

import importlib
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
_load = importlib.import_module("04_Load_ANN_TNN")
load_tnn_ternary = _load.load_tnn_ternary
print_model_summary = _load.print_model_summary





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






################################################################################
################################################################################
################################################################################
################################################################################
'''    REAL DEVICE (DEVICE-=LEVEL ONLY)  SONOS Vs APM programming error      '''
################################################################################
################################################################################
################################################################################
################################################################################




'''

if you just randomly any named it give you error like 


ValueError: Unknown programming error model: RAandomddsds.
Either define a model by that name or select from the11 existing options: 
    ['NormalIndependentDevice', 'NormalProportionalDevice', 
     'NormalInverseProportionalDevice', 'UniformIndependentDevice',
     'UniformProportionalDevice', 'UniformInverseProportionalDevice',
     'PCMJoshi', 'RRAMMilo', 'RRAMWan', 'SONOS', 'IdealDevice']
    


Based on file programming_modelin3.py I used this mdoel


first in cross-sim/simulator/devices/custom 
i create .py file which is APM_SINW_SBFET.py
and then also in devics/device.py i add
from .custom.APM_SINW_SBFET import APM_SINW_SBFET


also we can use SONOS and others too see the effects ....
Also we can go for some geenric read 


specify devcie errors (combination of them together)
why random reprogramming??

then we can add some ADC, DAC , Paraistics errors for getting worse 
to going for device-aware training

'''



PROGRAMMING_ERROR_MODELS = ("SONOS", "APM_SINW_SBFET")
PROG_ERROR_SEED = 42  # fixed seed per model build for reproducible comparison



def build_cs_params_tnn_real(programming_model: str) -> CrossSimParameters:
    """Balanced-core TNN CrossSim params with device programming error enabled."""
    p = CrossSimParameters()
    p.core.rows_max = 1024
    p.core.cols_max = 1024
    p.core.style = 1  # Balanced
    p.core.balanced.style = 1
    p.core.balanced.interleaved_posneg = False
    p.core.balanced.subtract_current_in_xbar = True
    p.core.weight_bits = 0
    p.core.bit_sliced.num_slices = 1
    p.xbar.device.Rmin = R_MIN
    p.xbar.device.Rmax = R_MAX
    p.xbar.device.cell_bits = 1
    p.xbar.device.Vread = 1.0
    p.xbar.device.programming_error.enable = True
    p.xbar.device.programming_error.model = programming_model
    p.xbar.device.read_noise.enable = False
    p.xbar.device.drift_error.enable = False
    p.xbar.adc.mvm.bits = 0
    p.xbar.adc.vmm.bits = 0
    p.xbar.dac.mvm.bits = 0
    p.xbar.dac.vmm.bits = 0
    p.xbar.array.parasitics.enable = False
    p.xbar.array.parasitics.Rp_row = 0
    p.xbar.array.parasitics.Rp_col = 0
    p.xbar.array.parasitics.Rp_row_terminal = 0
    p.xbar.array.parasitics.Rp_col_terminal = 0
    p.xbar.array.parasitics.current_from_input = True
    p.xbar.array.parasitics.selected_rows = "top"
    return p


convertible_modules(net_ann)
convertible_modules(net_tnn)

digital_real_ann_acc = evaluate(net_ann, test_loader, device="cpu")
digital_real_tnn_acc = evaluate(net_tnn, test_loader, device="cpu")

prog_error_nets: dict[str, dict[str, object]] = {}
prog_error_accuracy: dict[str, dict[str, float]] = {
    "ANN": {"Digital": digital_real_ann_acc},
    "TNN": {"Digital": digital_real_tnn_acc},
}

for prog_model in PROGRAMMING_ERROR_MODELS:
    np.random.seed(PROG_ERROR_SEED)

    cs_ann = build_cs_params_ann_real(prog_model)
    analog_ann = from_torch(net_ann, cs_ann)
    analog_ann.eval()
    ann_acc = evaluate(analog_ann, test_loader, device="cpu")

    np.random.seed(PROG_ERROR_SEED)

    cs_tnn = build_cs_params_tnn_real(prog_model)
    analog_tnn = from_torch(net_tnn, cs_tnn)
    analog_tnn.eval()
    tnn_acc = evaluate(analog_tnn, test_loader, device="cpu")

    prog_error_nets[prog_model] = {"ann": analog_ann, "tnn": analog_tnn}
    prog_error_accuracy["ANN"][prog_model] = ann_acc
    prog_error_accuracy["TNN"][prog_model] = tnn_acc

    print(f"\n=== Programming error model: {prog_model} ===")
    print(f"ANN  — Digital: {digital_real_ann_acc:.2f}%  |  CrossSim: {ann_acc:.2f}%")
    print(f"TNN  — Digital: {digital_real_tnn_acc:.2f}%  |  CrossSim: {tnn_acc:.2f}%")

'''
=== Programming error model: SONOS ===
ANN  — Digital: 98.00%  |  CrossSim: 96.88%
TNN  — Digital: 93.78%  |  CrossSim: 91.00%

=== Programming error model: APM_SINW_SBFET ===
ANN  — Digital: 98.00%  |  CrossSim: 97.94%
TNN  — Digital: 93.78%  |  CrossSim: 92.47%





After 25 may (new modeling)
=== Programming error model: SONOS ===
ANN  — Digital: 98.00%  |  CrossSim: 96.89%
TNN  — Digital: 93.78%  |  CrossSim: 91.05%

=== Programming error model: APM_SINW_SBFET ===
ANN  — Digital: 98.00%  |  CrossSim: 97.94%
TNN  — Digital: 93.78%  |  CrossSim: 92.56%


'''

# Backward-compatible handles (last entry = APM_SINW_SBFET)
cs_params_real_ternary = build_cs_params_tnn_real(PROGRAMMING_ERROR_MODELS[-1])
analog_real_tnn_net = prog_error_nets[PROGRAMMING_ERROR_MODELS[-1]]
analog_real_tnn_acc = prog_error_accuracy[PROGRAMMING_ERROR_MODELS[-1]]
digital_ideal_tnn_acc = digital_real_tnn_acc




# Programming-error model comparison: SONOS vs APM_SINW_SBFET


_COLOR_SONOS = "#2C5282"
_COLOR_APM = "#C05621"
_COLOR_DIGITAL_BAR = "#4A5568"
_PROG_MODEL_COLORS = {
    "SONOS": _COLOR_SONOS,
    "APM_SINW_SBFET": _COLOR_APM,
}
_PROG_MODEL_LABELS = {
    "SONOS": "SONOS",
    "APM_SINW_SBFET": "APM SiNW SBFET",
}
_LAYER_NAMES = ("fc1", "fc2", "fc3")



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




def _save_prog_fig(fig, stem: str, show: bool = True) -> None:
    if show:
        plt.show()
    else:
        plt.close(fig)


def _align_flat(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    n = min(a.size, b.size)
    return a[:n], b[:n]


def _layer_prog_stats(real_layers, ref_layers) -> list[dict]:
    """Per-layer statistics of programmed weights vs a reference (ideal or other model)."""
    stats = []
    n = min(len(real_layers), len(ref_layers))
    for i in range(n):
        w_real, w_ref = _align_flat(real_layers[i]["weight"], ref_layers[i]["weight"])
        delta = w_real - w_ref
        denom = np.maximum(np.abs(w_ref), 1e-12)
        stats.append({
            "layer_idx": i + 1,
            "layer_name": _LAYER_NAMES[i] if i < len(_LAYER_NAMES) else real_layers[i]["name"],
            "delta": delta,
            "rel_abs": np.abs(delta) / denom,
            "mae": float(np.mean(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(delta ** 2))),
            "corr": float(np.corrcoef(w_real, w_ref)[0, 1]),
        })
    return stats


def _collect_ann_bitslice_effective(model, normalize_by_Rmin: bool = True) -> list[dict]:
    """Per-layer, per-slice effective conductance (G_pos - G_neg) for bit-sliced ANN."""
    rows = []
    for layer_idx, li in enumerate(collect_bitsliced_analog_layers(model)):
        wc = li["wrapper_core"]
        scale = wc.params.xbar.device.Rmin if normalize_by_Rmin else 1.0
        for s in range(wc.Nslices):
            g_pos = np.array(wc.core_slices[s][0].matrix, dtype=np.float64) / scale
            g_neg = np.array(wc.core_slices[s][1].matrix, dtype=np.float64) / scale
            rows.append({
                "layer_idx": layer_idx + 1,
                "slice_idx": s,
                "g_pos": g_pos,
                "g_neg": g_neg,
                "g_eff": g_pos - g_neg,
            })
    return rows


def _collect_tnn_balanced_cores(model, normalize_by_Rmin: bool = True) -> list[dict]:
    """Per-layer positive / negative / effective conductance for balanced TNN."""
    rows = []
    for layer_idx, li in enumerate(collect_balanced_analog_layers(model)):
        wc = li["wrapper_core"]
        scale = wc.params.xbar.device.Rmin if normalize_by_Rmin else 1.0
        g_pos = np.array(wc.core_pos.matrix, dtype=np.float64) / scale
        g_neg = np.array(wc.core_neg.matrix, dtype=np.float64) / scale
        rows.append({
            "layer_idx": layer_idx + 1,
            "g_pos": g_pos,
            "g_neg": g_neg,
            "g_eff": g_pos - g_neg,
        })
    return rows









cs_params_sliced = CrossSimParameters()
cs_params_sliced.core.rows_max = 1024
cs_params_sliced.core.cols_max = 1024
cs_params_sliced.core.weight_bits =8
cs_params_sliced.core.style = 2 #Bitsliced
cs_params_sliced.core.bit_sliced.style = 1 #Bitsliced for each --> BALANCED
cs_params_sliced.core.bit_sliced.num_slices = 8
cs_params_sliced.xbar.device.Rmin = R_MIN  # LRS / ON
cs_params_sliced.xbar.device.Rmax = R_MAX  # HRS / OFF
cs_params_sliced.xbar.device.cell_bits = 1
cs_params_sliced.xbar.device.Vread = 1.0
cs_params_sliced.xbar.device.programming_error.enable = False
cs_params_sliced.xbar.device.read_noise.enable = False
cs_params_sliced.xbar.device.drift_error.enable = False
cs_params_sliced.xbar.adc.mvm.bits = 0
cs_params_sliced.xbar.adc.vmm.bits = 0
cs_params_sliced.xbar.dac.mvm.bits = 0
cs_params_sliced.xbar.dac.vmm.bits = 0
cs_params_sliced.xbar.array.parasitics.enable = False
cs_params_sliced.xbar.array.parasitics.Rp_row = 0
cs_params_sliced.xbar.array.parasitics.Rp_col = 0
cs_params_sliced.xbar.array.parasitics.Rp_row_terminal = 0
cs_params_sliced.xbar.array.parasitics.Rp_col_terminal = 0
cs_params_sliced.xbar.array.parasitics.current_from_input = True
cs_params_sliced.xbar.array.parasitics.selected_rows = "top"

#convert pytorch to crossim analog model
analog_ideal_ann_net = from_torch(net_ann, cs_params_sliced)
analog_ideal_ann_net.eval()





cs_params_ternary = CrossSimParameters()
cs_params_ternary.core.rows_max = 1024
cs_params_ternary.core.cols_max = 1024
cs_params_ternary.core.style = 1  # CoreStyle.BALANCED
cs_params_ternary.core.balanced.style = 1
cs_params_ternary.core.balanced.interleaved_posneg = False
cs_params_ternary.core.balanced.subtract_current_in_xbar = True
cs_params_ternary.core.weight_bits = 0
cs_params_ternary.core.bit_sliced.num_slices = 1
cs_params_ternary.xbar.device.Rmin = R_MIN  # LRS / ON
cs_params_ternary.xbar.device.Rmax = R_MAX  # HRS / OFF
cs_params_ternary.xbar.device.cell_bits = 1
cs_params_ternary.xbar.device.Vread = 1.0
cs_params_ternary.xbar.device.programming_error.enable = False
cs_params_ternary.xbar.device.read_noise.enable = False
cs_params_ternary.xbar.device.drift_error.enable = False
cs_params_ternary.xbar.adc.mvm.bits = 0
cs_params_ternary.xbar.adc.vmm.bits = 0
cs_params_ternary.xbar.dac.mvm.bits = 0
cs_params_ternary.xbar.dac.vmm.bits = 0
cs_params_ternary.xbar.array.parasitics.enable = False
cs_params_ternary.xbar.array.parasitics.Rp_row = 0
cs_params_ternary.xbar.array.parasitics.Rp_col = 0
cs_params_ternary.xbar.array.parasitics.Rp_row_terminal = 0
cs_params_ternary.xbar.array.parasitics.Rp_col_terminal = 0
cs_params_ternary.xbar.array.parasitics.current_from_input = True
cs_params_ternary.xbar.array.parasitics.selected_rows = "top"

#convert pytorch to crossim analog model
analog_ideal_tnn_net = from_torch(net_tnn, cs_params_ternary)
analog_ideal_tnn_net.eval()














# --- Reference layers (ideal CrossSim, no programming error) ---
ideal_ann_ref_layers = collect_analog_layers(analog_ideal_ann_net)
ideal_tnn_ref_layers = collect_analog_layers(analog_ideal_tnn_net)
ideal_ann_bitslices = _collect_ann_bitslice_effective(analog_ideal_ann_net)
ideal_tnn_balanced = _collect_tnn_balanced_cores(analog_ideal_tnn_net)

prog_layers = {}
prog_vs_ideal = {}
prog_ann_slices = {}
prog_tnn_balanced = {}

for model_name in PROGRAMMING_ERROR_MODELS:
    ann_net = prog_error_nets[model_name]["ann"]
    tnn_net = prog_error_nets[model_name]["tnn"]
    prog_layers[model_name] = {
        "ann": collect_analog_layers(ann_net),
        "tnn": collect_analog_layers(tnn_net),
    }
    prog_vs_ideal[model_name] = {
        "ann": _layer_prog_stats(prog_layers[model_name]["ann"], ideal_ann_ref_layers),
        "tnn": _layer_prog_stats(prog_layers[model_name]["tnn"], ideal_tnn_ref_layers),
    }
    prog_ann_slices[model_name] = _collect_ann_bitslice_effective(ann_net)
    prog_tnn_balanced[model_name] = _collect_tnn_balanced_cores(tnn_net)

sonos_key, apm_key = PROGRAMMING_ERROR_MODELS

# Cross-model layer stats (SONOS programmed weights vs APM programmed weights)
cross_model_ann = _layer_prog_stats(
    prog_layers[sonos_key]["ann"], prog_layers[apm_key]["ann"]
)
cross_model_tnn = _layer_prog_stats(
    prog_layers[sonos_key]["tnn"], prog_layers[apm_key]["tnn"]
)

print("\n" + "=" * 72)
print("Programming-error comparison summary (vs ideal CrossSim mapping)")
print("=" * 72)
for net_label in ("ANN", "TNN"):
    print(f"\n{net_label}:")
    for model_name in PROGRAMMING_ERROR_MODELS:
        acc = prog_error_accuracy[net_label][model_name]
        drop = prog_error_accuracy[net_label]["Digital"] - acc
        print(f"  {_PROG_MODEL_LABELS[model_name]:16s}  acc={acc:5.2f}%  Δdigital={drop:+.2f} pp")
        for st in prog_vs_ideal[model_name][net_label.lower()]:
            print(
                f"    {st['layer_name']:4s}  MAE={st['mae']:.4e}  "
                f"RMSE={st['rmse']:.4e}  r(ideal)={st['corr']:.4f}"
            )
            
            

print("\nCross-model effective-weight correlation (SONOS vs APM):")
for label, stats in (("ANN", cross_model_ann), ("TNN", cross_model_tnn)):
    for st in stats:
        print(f"  {label} {st['layer_name']}: r={st['corr']:.4f}  MAE={st['mae']:.4e}")
        
        


'''
========================================================================
Programming-error comparison summary (vs ideal CrossSim mapping)
========================================================================

ANN:
  SONOS             acc=96.89%  Δdigital=+1.11 pp
    fc1   MAE=3.5337e-02  RMSE=4.4269e-02  r(ideal)=0.7781
    fc2   MAE=4.0178e-02  RMSE=5.0437e-02  r(ideal)=0.8837
    fc3   MAE=4.1116e-02  RMSE=5.1480e-02  r(ideal)=0.9570
  APM SiNW SBFET    acc=97.94%  Δdigital=+0.06 pp
    fc1   MAE=1.6718e-02  RMSE=2.0900e-02  r(ideal)=0.9400
    fc2   MAE=1.8903e-02  RMSE=2.3636e-02  r(ideal)=0.9735
    fc3   MAE=1.9201e-02  RMSE=2.3860e-02  r(ideal)=0.9921

TNN:
  SONOS             acc=91.05%  Δdigital=+2.73 pp
    fc1   MAE=7.0239e-02  RMSE=9.4222e-02  r(ideal)=0.9687
    fc2   MAE=7.8196e-02  RMSE=1.0671e-01  r(ideal)=0.9760
    fc3   MAE=1.0574e-01  RMSE=1.4184e-01  r(ideal)=0.9756
  APM SiNW SBFET    acc=92.56%  Δdigital=+1.22 pp
    fc1   MAE=3.0888e-02  RMSE=3.8551e-02  r(ideal)=0.9959
    fc2   MAE=3.2878e-02  RMSE=4.0907e-02  r(ideal)=0.9975
    fc3   MAE=4.0606e-02  RMSE=4.9058e-02  r(ideal)=0.9987

Cross-model effective-weight correlation (SONOS vs APM):
  ANN fc1: r=0.7310  MAE=3.9094e-02
  ANN fc2: r=0.8609  MAE=4.4213e-02
  ANN fc3: r=0.9505  MAE=4.5141e-02
  TNN fc1: r=0.9646  MAE=7.6762e-02
  TNN fc2: r=0.9737  MAE=8.4361e-02
  TNN fc3: r=0.9744  MAE=1.1451e-01
  
  
'''





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

# --- Figure 1: MNIST test accuracy ---
with plt.rc_context(_ACADEMIC_RC):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    x = np.arange(len(PROGRAMMING_ERROR_MODELS))
    width = 0.32

    for ax, net_label in zip(axes, ("ANN", "TNN")):
        digital_acc = prog_error_accuracy[net_label]["Digital"]
        accs = [prog_error_accuracy[net_label][m] for m in PROGRAMMING_ERROR_MODELS]
        colors = [_PROG_MODEL_COLORS[m] for m in PROGRAMMING_ERROR_MODELS]

        ax.axhline(
            digital_acc, color=_COLOR_DIGITAL_BAR, linestyle="--", linewidth=1.2,
            label=f"Digital ({digital_acc:.2f}%)", zorder=1,
        )
        bars = ax.bar(
            x, accs, width=0.55, color=colors,
            edgecolor="white", linewidth=0.8, zorder=3,
        )
        ax.set_xticks(x, [_PROG_MODEL_LABELS[m] for m in PROGRAMMING_ERROR_MODELS])
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title(f"{net_label}: Programming-Error Model Comparison")
        y0 = min(digital_acc, min(accs)) - 1.5
        y1 = max(digital_acc, max(accs)) + 0.8
        ax.set_ylim(y0, y1)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.45, linestyle="--", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, acc in zip(bars, accs):
            ax.text(
                bar.get_x() + bar.get_width() / 2, acc + 0.05,
                f"{acc:.2f}%", ha="center", va="bottom", fontsize=9,
            )

    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle(
        "SONOS vs APM SiNW SBFET — MNIST Inference Accuracy\n"
        "(programming error only; read noise and drift disabled)",
        y=1.02, fontsize=12,
    )
    fig.tight_layout()
    _save_prog_fig(fig, "fig01_accuracy_sonos_vs_apm")


# --- Figure 2: Accuracy drop from digital baseline ---
with plt.rc_context(_ACADEMIC_RC):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    nets = ["ANN", "TNN"]
    x = np.arange(len(nets))
    width = 0.35

    sonos_drop = [
        prog_error_accuracy[n]["Digital"] - prog_error_accuracy[n][sonos_key]
        for n in nets
    ]
    apm_drop = [
        prog_error_accuracy[n]["Digital"] - prog_error_accuracy[n][apm_key]
        for n in nets
    ]

    ax.bar(x - width / 2, sonos_drop, width, label=_PROG_MODEL_LABELS[sonos_key],
           color=_COLOR_SONOS, edgecolor="white", linewidth=0.8)
    ax.bar(x + width / 2, apm_drop, width, label=_PROG_MODEL_LABELS[apm_key],
           color=_COLOR_APM, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x, nets)
    ax.set_ylabel("Accuracy Drop from Digital (pp)")
    ax.set_title("Impact of Programming Error on MNIST Accuracy")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.45, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left")
    for i, (ds, da) in enumerate(zip(sonos_drop, apm_drop)):
        ax.text(i - width / 2, ds + 0.02, f"{ds:.2f}", ha="center", fontsize=9)
        ax.text(i + width / 2, da + 0.02, f"{da:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    _save_prog_fig(fig, "fig02_accuracy_drop_pp")


# --- Figure 3 & 4: SONOS vs APM effective-weight scatter (per layer) ---
def _plot_crossmodel_scatter(
    stats_list, net_label: str, xlabel_model: str, ylabel_model: str, stem: str,
) -> None:
    n = len(stats_list)
    with plt.rc_context(_ACADEMIC_RC):
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)
        axes = axes[0]
        for ax, st in zip(axes, stats_list):
            w_x, w_y = _align_flat(
                prog_layers[sonos_key][net_label.lower()][st["layer_idx"] - 1]["weight"],
                prog_layers[apm_key][net_label.lower()][st["layer_idx"] - 1]["weight"],
            )
            ax.scatter(
                w_x, w_y, s=3, alpha=0.25, color=_COLOR_SONOS,
                edgecolors="none", rasterized=True,
            )
            lims = [
                min(w_x.min(), w_y.min()), max(w_x.max(), w_y.max()),
            ]
            ax.plot(lims, lims, "k--", linewidth=0.9, alpha=0.7)
            ax.set_xlabel(f"{xlabel_model} weight")
            ax.set_ylabel(f"{ylabel_model} weight")
            ax.set_title(f"{st['layer_name']}\n$r$={st['corr']:.4f}, MAE={st['mae']:.2e}")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.4, linestyle="--", linewidth=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.suptitle(
            f"{net_label}: Programmed Effective Weights — "
            f"{_PROG_MODEL_LABELS[sonos_key]} vs {_PROG_MODEL_LABELS[apm_key]}",
            y=1.02,
        )
        fig.tight_layout()
        _save_prog_fig(fig, stem)


_plot_crossmodel_scatter(
    cross_model_ann, "ANN", _PROG_MODEL_LABELS[sonos_key],
    _PROG_MODEL_LABELS[apm_key], "fig03_ann_crossmodel_weight_scatter",
)
_plot_crossmodel_scatter(
    cross_model_tnn, "TNN", _PROG_MODEL_LABELS[sonos_key],
    _PROG_MODEL_LABELS[apm_key], "fig04_tnn_crossmodel_weight_scatter",
)


# --- Figure 5 & 6: Programming displacement vs ideal mapping (boxplots) ---
def _plot_prog_displacement_boxplots(vs_ideal_dict, net_label: str, stem: str) -> None:
    n_layers = len(vs_ideal_dict[sonos_key])
    with plt.rc_context(_ACADEMIC_RC):
        fig, axes = plt.subplots(1, n_layers, figsize=(4.0 * n_layers, 4.5), sharey=True)
        if n_layers == 1:
            axes = [axes]
        for ax, layer_i in zip(axes, range(n_layers)):
            data, labels, colors = [], [], []
            for model_name in PROGRAMMING_ERROR_MODELS:
                rel = vs_ideal_dict[model_name][layer_i]["rel_abs"]
                data.append(rel[np.isfinite(rel)])
                labels.append(_PROG_MODEL_LABELS[model_name])
                colors.append(_PROG_MODEL_COLORS[model_name])
            bp = ax.boxplot(
                data, tick_labels=labels, patch_artist=True, showfliers=False,
                medianprops=dict(color="black", linewidth=1.2),
            )
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.55)
            ax.set_yscale("log")
            ax.set_title(_LAYER_NAMES[layer_i])
            ax.set_xlabel("Programming-error model")
            ax.grid(axis="y", alpha=0.4, linestyle="--", linewidth=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[0].set_ylabel(r"$|G_\mathrm{prog} - G_\mathrm{ideal}| / |G_\mathrm{ideal}|$")
        fig.suptitle(
            f"{net_label}: Relative Programming Displacement vs Ideal CrossSim",
            y=1.02,
        )
        fig.tight_layout()
        _save_prog_fig(fig, stem)


_plot_prog_displacement_boxplots(
    {m: prog_vs_ideal[m]["ann"] for m in PROGRAMMING_ERROR_MODELS},
    "ANN",
    "fig05_ann_prog_displacement_vs_ideal",
)
_plot_prog_displacement_boxplots(
    {m: prog_vs_ideal[m]["tnn"] for m in PROGRAMMING_ERROR_MODELS},
    "TNN",
    "fig06_tnn_prog_displacement_vs_ideal",
)


def _slice_mae_heatmap(slice_rows, ref_rows, key: str = "g_eff") -> np.ndarray:
    """Layer × bit-slice matrix of mean |ΔG| vs ideal reference."""
    ref_map = {(r["layer_idx"], r["slice_idx"]): r[key] for r in ref_rows}
    n_layers = max(r["layer_idx"] for r in slice_rows)
    n_slices = max(r["slice_idx"] for r in slice_rows) + 1
    mat = np.full((n_layers, n_slices), np.nan)
    for row in slice_rows:
        k = (row["layer_idx"], row["slice_idx"])
        if k not in ref_map:
            continue
        a, b = _align_flat(row[key], ref_map[k])
        mat[row["layer_idx"] - 1, row["slice_idx"]] = np.mean(np.abs(a - b))
    return mat


# --- Figure 7: ANN bit-slice MAE vs ideal (heatmaps) ---
slice_mae_sonos = _slice_mae_heatmap(prog_ann_slices[sonos_key], ideal_ann_bitslices)
slice_mae_apm = _slice_mae_heatmap(prog_ann_slices[apm_key], ideal_ann_bitslices)
slice_mae_diff = slice_mae_apm - slice_mae_sonos

with plt.rc_context(_ACADEMIC_RC):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    vmax = np.nanmax(np.stack([slice_mae_sonos, slice_mae_apm]))
    ims = []
    for ax, data, title in zip(
        axes,
        (slice_mae_sonos, slice_mae_apm, slice_mae_diff),
        (
            f"{_PROG_MODEL_LABELS[sonos_key]}",
            f"{_PROG_MODEL_LABELS[apm_key]}",
            "APM − SONOS",
        ),
    ):
        if title == "APM − SONOS":
            im = ax.imshow(
                data, aspect="auto", cmap="RdBu_r",
                vmin=-np.nanmax(np.abs(data)), vmax=np.nanmax(np.abs(data)),
            )
        else:
            im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        ims.append(im)
        ax.set_xticks(range(data.shape[1]), labels=[str(i) for i in range(data.shape[1])])
        ax.set_yticks(
            range(data.shape[0]),
            labels=[_LAYER_NAMES[i] for i in range(data.shape[0])],
        )
        ax.set_xlabel("Bit slice")
        ax.set_ylabel("Layer")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"MAE ($G/R_\mathrm{min}$)")
    fig.suptitle(
        "ANN: Mean Absolute Programming Error per Layer and Bit Slice\n"
        "(effective conductance $G_+ - G_-$ vs ideal mapping)",
        y=1.05,
    )
    fig.tight_layout()
    _save_prog_fig(fig, "fig07_ann_bitslice_mae_heatmap")


# --- Figure 8: TNN positive / negative core MAE vs ideal ---
def _get_ann_slice_row(slice_list: list[dict], layer_idx: int, slice_idx: int) -> dict | None:
    for row in slice_list:
        if row["layer_idx"] == layer_idx and row["slice_idx"] == slice_idx:
            return row
    return None


def _tnn_core_mae_bars() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_labels, mae_pos_s, mae_neg_s, mae_pos_a, mae_neg_a = [], [], [], [], []
    for i, (row_s, row_a, row_i) in enumerate(
        zip(prog_tnn_balanced[sonos_key], prog_tnn_balanced[apm_key], ideal_tnn_balanced)
    ):
        x_labels.append(_LAYER_NAMES[i])
        gp_s, gp_i = _align_flat(row_s["g_pos"], row_i["g_pos"])
        gn_s, gn_i = _align_flat(row_s["g_neg"], row_i["g_neg"])
        gp_a, _ = _align_flat(row_a["g_pos"], row_i["g_pos"])
        gn_a, _ = _align_flat(row_a["g_neg"], row_i["g_neg"])
        mae_pos_s.append(np.mean(np.abs(gp_s - gp_i)))
        mae_neg_s.append(np.mean(np.abs(gn_s - gn_i)))
        mae_pos_a.append(np.mean(np.abs(gp_a - gp_i)))
        mae_neg_a.append(np.mean(np.abs(gn_a - gn_i)))
    return (
        x_labels,
        np.array(mae_pos_s), np.array(mae_neg_s),
        np.array(mae_pos_a), np.array(mae_neg_a),
    )


x_labels, mae_pos_s, mae_neg_s, mae_pos_a, mae_neg_a = _tnn_core_mae_bars()
x = np.arange(len(x_labels))
width = 0.18

with plt.rc_context(_ACADEMIC_RC):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - 1.5 * width, mae_pos_s, width, label=f"{_PROG_MODEL_LABELS[sonos_key]} (+)",
           color=_COLOR_SONOS, alpha=0.9, edgecolor="white")
    ax.bar(x - 0.5 * width, mae_neg_s, width, label=f"{_PROG_MODEL_LABELS[sonos_key]} (−)",
           color=_COLOR_SONOS, alpha=0.45, edgecolor="white", hatch="//")
    ax.bar(x + 0.5 * width, mae_pos_a, width, label=f"{_PROG_MODEL_LABELS[apm_key]} (+)",
           color=_COLOR_APM, alpha=0.9, edgecolor="white")
    ax.bar(x + 1.5 * width, mae_neg_a, width, label=f"{_PROG_MODEL_LABELS[apm_key]} (−)",
           color=_COLOR_APM, alpha=0.45, edgecolor="white", hatch="//")
    ax.set_xticks(x, x_labels)
    ax.set_ylabel(r"MAE vs ideal ($G/R_\mathrm{min}$)")
    ax.set_xlabel("Layer")
    ax.set_title("TNN Balanced Core: Programming Error on Positive and Negative Arrays")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.45, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save_prog_fig(fig, "fig08_tnn_posneg_mae_vs_ideal")


# --- Figure 9: Cross-model correlation per layer (SONOS vs APM) ---
with plt.rc_context(_ACADEMIC_RC):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, stats, net_label in zip(
        axes, (cross_model_ann, cross_model_tnn), ("ANN", "TNN"),
    ):
        layers = [st["layer_name"] for st in stats]
        corrs = [st["corr"] for st in stats]
        x = np.arange(len(layers))
        ax.bar(x, corrs, color=_COLOR_SONOS, edgecolor="white", linewidth=0.8, width=0.55)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xticks(x, layers)
        ax.set_ylim(0.70, 1.01)
        ax.set_title(net_label)
        ax.set_xlabel("Layer")
        for i, c in enumerate(corrs):
            ax.text(i, c + 0.00015, f"{c:.4f}", ha="center", fontsize=8)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.45, linestyle="--", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(r"Pearson $r$ (SONOS vs APM weights)")
    fig.suptitle("Cross-Model Weight Correlation by Layer", y=1.02)
    fig.tight_layout()
    _save_prog_fig(fig, "fig09_crossmodel_correlation_by_layer")


# --- Figure 10: Pooled cross-model programming-error distribution ---
pooled_delta_ann = np.concatenate([st["delta"] for st in cross_model_ann])
pooled_delta_tnn = np.concatenate([st["delta"] for st in cross_model_tnn])

with plt.rc_context(_ACADEMIC_RC):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, delta, title in zip(
        axes,
        (pooled_delta_ann, pooled_delta_tnn),
        ("ANN", "TNN"),
    ):
        ax.hist(
            delta, bins=80, density=True, alpha=0.75,
            color=_COLOR_SONOS, edgecolor="white", linewidth=0.4,
        )
        ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel(r"$\Delta w = w_\mathrm{APM} - w_\mathrm{SONOS}$")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.set_axisbelow(True)
        ax.grid(alpha=0.4, linestyle="--", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            0.97, 0.95,
            f"MAE={np.mean(np.abs(delta)):.2e}\nstd={delta.std():.2e}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#CCC"),
        )
    fig.suptitle(
        "Distribution of Programmed Weight Differences (APM − SONOS)",
        y=1.02,
    )
    fig.tight_layout()
    _save_prog_fig(fig, "fig10_crossmodel_delta_histogram")


# --- Figure 11: ANN per-slice SONOS vs APM scatter (slice 0 and 7 exemplars) ---
_exemplar_slices = (0, 7)
with plt.rc_context(_ACADEMIC_RC):
    fig, axes = plt.subplots(
        len(_exemplar_slices), 3, figsize=(12, 3.6 * len(_exemplar_slices)),
    )
    for row, sl in enumerate(_exemplar_slices):
        for col, layer_i in enumerate(range(3)):
            ax = axes[row, col]
            rs = _get_ann_slice_row(prog_ann_slices[sonos_key], layer_i + 1, sl)
            ra = _get_ann_slice_row(prog_ann_slices[apm_key], layer_i + 1, sl)
            if rs is None or ra is None:
                ax.set_visible(False)
                continue
            gx, gy = _align_flat(rs["g_eff"], ra["g_eff"])
            r_val = float(np.corrcoef(gx, gy)[0, 1])
            ax.scatter(gx, gy, s=2, alpha=0.3, color=_COLOR_SONOS, edgecolors="none", rasterized=True)
            lim = max(abs(gx).max(), abs(gy).max()) * 1.05
            ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.6)
            ax.set_title(f"{_LAYER_NAMES[layer_i]}, slice {sl}\n$r$={r_val:.4f}")
            ax.set_xlabel("SONOS $G_\\mathrm{eff}$")
            if col == 0:
                ax.set_ylabel("APM $G_\\mathrm{eff}$")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.suptitle(
        "ANN Bit Slices: Effective Conductance SONOS vs APM (slices 0 and 7)",
        y=1.02,
    )
    fig.tight_layout()
    _save_prog_fig(fig, "fig11_ann_bitslice_scatter_exemplars")























################################################################################
################################################################################
################################################################################
################################################################################
'''    Hardware cost (TNN ternary balanced)     '''
################################################################################
################################################################################
################################################################################
################################################################################




#-------------------------------------------------------------------------------
# Hardware cost: ANN (bit-sliced) vs TNN (ternary balanced)
# Memristor / crossbar counts from actual layer weight shapes

ANN_NUM_SLICES=8

HW_ANN_SLICES = ANN_NUM_SLICES  # 8-bit bit-sliced → 8 slices
HW_BALANCED_SIDES = 2           # G_pos and G_neg per balanced slice/array
HW_LAYER_NAMES = ("fc1", "fc2", "fc3")


def _save_hw_fig(fig, stem: str) -> None:
    #fig.savefig(FIGURES_HW_DIR / f"{stem}.pdf", bbox_inches="tight", dpi=300)
    #fig.savefig(FIGURES_HW_DIR / f"{stem}.png", bbox_inches="tight", dpi=FIGURE_DPI)
    plt.show()


def _linear_weight_shape(model, layer_name: str) -> tuple[int, int]:
    w = getattr(model, layer_name).weight
    return int(w.shape[0]), int(w.shape[1])  # (out_features, in_features)


def _count_ann_hardware(out: int, inp: int) -> dict:
    """Bit-sliced balanced: N_slices × (G_pos + G_neg), one device per matrix entry each."""
    synapses = out * inp
    crossbars = HW_ANN_SLICES * HW_BALANCED_SIDES
    memristors = synapses * crossbars
    adcs_per_layer = HW_ANN_SLICES
    return {
        "out": out,
        "in": inp,
        "synapses": synapses,
        "crossbars": crossbars,
        "memristors": memristors,
        "adcs": adcs_per_layer,
    }


def _count_tnn_hardware(out: int, inp: int, weight: np.ndarray) -> dict:
    """Balanced ternary: G_pos + G_neg; report non-zero ternary occupancy."""
    synapses = out * inp
    crossbars = HW_BALANCED_SIDES
    memristors = synapses * crossbars
    w_flat = weight.ravel()
    n_total = w_flat.size
    n_pos = int((w_flat > 0).sum())
    n_neg = int((w_flat < 0).sum())
    n_zero = int((w_flat == 0).sum())
    return {
        "out": out,
        "in": inp,
        "synapses": synapses,
        "crossbars": crossbars,
        "memristors": memristors,
        "adcs": 1,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_zero": n_zero,
        "n_total": n_total,
        "pct_nonzero": 100.0 * (n_pos + n_neg) / n_total if n_total else 0.0,
    }


def collect_layer_info(model) -> list[dict]:
    """Extract weight/bias arrays per Linear layer."""
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


ann_hw_layers = []
tnn_hw_layers = []
ann_layer_info = collect_layer_info(net_ann)
tnn_layer_info = collect_layer_info(net_tnn)

for name, ann_li, tnn_li in zip(HW_LAYER_NAMES, ann_layer_info, tnn_layer_info):
    out_a, in_a = _linear_weight_shape(net_ann, name)
    out_t, in_t = _linear_weight_shape(net_tnn, name)
    ann_hw_layers.append({"name": name, **_count_ann_hardware(out_a, in_a)})
    tnn_hw_layers.append({
        "name": name,
        **_count_tnn_hardware(out_t, in_t, tnn_li["weight"]),
    })

ann_totals = {k: sum(L[k] for L in ann_hw_layers)
              for k in ("synapses", "crossbars", "memristors", "adcs")}
tnn_totals = {k: sum(L[k] for L in tnn_hw_layers)
              for k in ("synapses", "crossbars", "memristors", "adcs")}

print("\n" + "=" * 78)
print("Hardware cost — ANN (8-bit bit-sliced balanced) vs TNN (ternary balanced)")
print("=" * 78)
print(
    f"{'Layer':<6} {'Shape':>12}  "
    f"{'ANN memristors':>16}  {'TNN memristors':>16}  {'Ratio ANN/TNN':>14}"
)
print("-" * 78)
for a, t in zip(ann_hw_layers, tnn_hw_layers):
    shape = f"{a['out']}×{a['in']}"
    ratio = a["memristors"] / t["memristors"] if t["memristors"] else float("inf")
    print(
        f"{a['name']:<6} {shape:>12}  "
        f"{a['memristors']:>16,}  {t['memristors']:>16,}  {ratio:>14.1f}×"
    )
print("-" * 78)
total_ratio = ann_totals["memristors"] / tnn_totals["memristors"]
print(
    f"{'TOTAL':<6} {'':<12}  "
    f"{ann_totals['memristors']:>16,}  {tnn_totals['memristors']:>16,}  {total_ratio:>14.1f}×"
)
print(
    f"\nCrossbar sub-arrays (pos/neg planes): ANN {ann_totals['crossbars']}  |  "
    f"TNN {tnn_totals['crossbars']}  (ratio {ann_totals['crossbars'] / tnn_totals['crossbars']:.1f}×)"
)
print(
    f"ADC blocks (per layer, 8b RampADC model): ANN {ann_totals['adcs']}  |  "
    f"TNN {tnn_totals['adcs']}"
)
print("\nTNN ternary weight occupancy (from checkpoint weights):")
for t in tnn_hw_layers:
    print(
        f"  {t['name']} ({t['out']}×{t['in']}):  "
        f"+1: {t['n_pos']:,}  -1: {t['n_neg']:,}  0: {t['n_zero']:,}  "
        f"({t['pct_nonzero']:.1f}% non-zero)"
    )
print(
    f"\nEncoding: ANN uses {HW_ANN_SLICES} bit slices × {HW_BALANCED_SIDES} "
    f"(G+/G−) = {HW_ANN_SLICES * HW_BALANCED_SIDES}× devices per synapse; "
    f"TNN uses {HW_BALANCED_SIDES}× (G+/G−) per synapse."
)


'''
==============================================================================
Hardware cost — ANN (8-bit bit-sliced balanced) vs TNN (ternary balanced)
==============================================================================
Layer         Shape    ANN memristors    TNN memristors   Ratio ANN/TNN
------------------------------------------------------------------------------
fc1         256×784         3,211,264           401,408             8.0×
fc2         128×256           524,288            65,536             8.0×
fc3          10×128            20,480             2,560             8.0×
------------------------------------------------------------------------------
TOTAL                       3,756,032           469,504             8.0×

Crossbar sub-arrays (pos/neg planes): ANN 48  |  TNN 6  (ratio 8.0×)
ADC blocks (per layer, 8b RampADC model): ANN 24  |  TNN 3

TNN ternary weight occupancy (from checkpoint weights):
  fc1 (256×784):  +1: 16,621  -1: 10,772  0: 173,311  (13.6% non-zero)
  fc2 (128×256):  +1: 2,350  -1: 5,365  0: 25,053  (23.5% non-zero)
  fc3 (10×128):  +1: 102  -1: 700  0: 478  (62.7% non-zero)

Encoding: ANN uses 8 bit slices × 2 (G+/G−) = 16× devices per synapse; TNN uses 2× (G+/G−) per synapse.


'''







layer_labels = [L["name"] for L in ann_hw_layers]
x = np.arange(len(layer_labels))
ann_mem = [L["memristors"] for L in ann_hw_layers]
tnn_mem = [L["memristors"] for L in tnn_hw_layers]
ann_xbars = [L["crossbars"] for L in ann_hw_layers]
tnn_xbars = [L["crossbars"] for L in tnn_hw_layers]


_COLOR_ANN_STRAT = "#2C5282"
_COLOR_TNN_STRAT = "#C05621"





# --- Figure 0: Mapping strategy comparison under APM programming error ---
def plot_mapping_strategy_comparison():
    
    ann_digital = prog_error_accuracy["ANN"]["Digital"]
    tnn_digital = prog_error_accuracy["TNN"]["Digital"]

    ann_ideal = evaluate(analog_ideal_ann_net, test_loader, device="cpu")
    tnn_ideal = evaluate(analog_ideal_tnn_net, test_loader, device="cpu")

    ann_apm = prog_error_accuracy["ANN"]["APM_SINW_SBFET"]
    tnn_apm = prog_error_accuracy["TNN"]["APM_SINW_SBFET"]

    groups = [
        "ANN\n(8-bit bit-sliced)",
        "TNN\n(ternary balanced)"
    ]

    digital_vals = [ann_digital, tnn_digital]
    ideal_vals   = [ann_ideal, tnn_ideal]
    apm_vals     = [ann_apm, tnn_apm]

    x = np.arange(len(groups))
    width = 0.24

    COLOR_DIGITAL = "#355C8A"
    COLOR_IDEAL   = "#7A889F"
    COLOR_APM     = "#C15A22"

    with plt.rc_context(_ACADEMIC_RC):

        fig, ax = plt.subplots(figsize=(10, 5.6))

        bars1 = ax.bar(
            x - width,
            digital_vals,
            width,
            label="Digital PyTorch",
            color=COLOR_DIGITAL,
            edgecolor="white",
            linewidth=0.8,
        )

        bars2 = ax.bar(
            x,
            ideal_vals,
            width,
            label="Ideal CrossSim",
            color=COLOR_IDEAL,
            edgecolor="white",
            linewidth=0.8,
        )

        bars3 = ax.bar(
            x + width,
            apm_vals,
            width,
            label="APM SiNW SBFET",
            color=COLOR_APM,
            edgecolor="white",
            linewidth=0.8,
        )

        ax.set_xticks(x, groups)

        ax.set_ylabel("MNIST Test Accuracy (%)")

        ax.set_title(
            "Mapping Strategy Comparison under APM SiNW SBFET Programming Error\n"
            "ANN (bit-sliced) vs TNN (ternary balanced)"
        )

        y_min = min(digital_vals + ideal_vals + apm_vals) - 4
        y_max = max(digital_vals + ideal_vals + apm_vals) + 1

        ax.set_ylim(y_min, y_max)

        ax.grid(
            axis="y",
            linestyle="--",
            linewidth=0.6,
            alpha=0.4,
        )

        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        def _annotate(bars):
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    h + 0.12,
                    f"{h:.2f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        _annotate(bars1)
        _annotate(bars2)
        _annotate(bars3)

        legend = ax.legend(loc="upper right")
        legend.get_frame().set_alpha(0.95)

        fig.tight_layout()

        _save_prog_fig(fig, "fig00_mapping_strategy_comparison")


plot_mapping_strategy_comparison()




# --- Figure HW-2: Crossbar sub-arrays per layer ---
with plt.rc_context(_ACADEMIC_RC):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width / 2, ann_xbars, width, label="ANN", color=_COLOR_ANN_STRAT,
           edgecolor="white", linewidth=0.8)
    ax.bar(x + width / 2, tnn_xbars, width, label="TNN", color=_COLOR_TNN_STRAT,
           edgecolor="white", linewidth=0.8)
    ax.set_xticks(x, layer_labels)
    ax.set_ylabel("Crossbar sub-arrays (G$_+$ or G$_-$ planes)")
    ax.set_xlabel("Layer")
    ax.set_title("Number of Crossbar Arrays to Fabricate per Layer")
    ax.legend(loc="upper right")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.45, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save_hw_fig(fig, "fig02_crossbars_per_layer")


# --- Figure HW-3: Memristors per layer ---
with plt.rc_context(_ACADEMIC_RC):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    width = 0.35
    ax.bar(x - width / 2, ann_mem, width, label="ANN (8-bit bit-sliced)",
           color=_COLOR_ANN_STRAT, edgecolor="white", linewidth=0.8)
    ax.bar(x + width / 2, tnn_mem, width, label="TNN (ternary balanced)",
           color=_COLOR_TNN_STRAT, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x, layer_labels)
    ax.set_ylabel("Memristor devices (count)")
    ax.set_xlabel("Layer")
    ax.set_title(
        "Physical Memristor Count per Layer\n"
        f"(ANN: {HW_ANN_SLICES} slices × G$_+$/G$_-$; TNN: G$_+$/G$_-$ only)",
    )
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v/1e6:.2f}M" if v >= 1e6 else f"{v/1e3:.0f}k" if v >= 1e3 else f"{v:.0f}")
    )
    ax.grid(axis="y", alpha=0.45, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right")
    for i, (am, tm) in enumerate(zip(ann_mem, tnn_mem)):
        ax.text(i - width / 2, am, f"{am/1e6:.2f}M", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, tm, f"{tm/1e3:.0f}k", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    _save_hw_fig(fig, "fig01_memristors_per_layer")


# --- Figure HW-4: System totals (memristors, crossbars, ADCs) ---
categories = ["Memristors", "Crossbar\nsub-arrays", "ADC\nblocks"]
ann_tot_vals = [ann_totals["memristors"], ann_totals["crossbars"], ann_totals["adcs"]]
tnn_tot_vals = [tnn_totals["memristors"], tnn_totals["crossbars"], tnn_totals["adcs"]]
x_cat = np.arange(len(categories))

with plt.rc_context(_ACADEMIC_RC):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(x_cat - width / 2, ann_tot_vals, width, label="ANN", color=_COLOR_ANN_STRAT,
           edgecolor="white", linewidth=0.8)
    ax.bar(x_cat + width / 2, tnn_tot_vals, width, label="TNN", color=_COLOR_TNN_STRAT,
           edgecolor="white", linewidth=0.8)
    ax.set_xticks(x_cat, categories)
    ax.set_ylabel("Count (entire network)")
    ax.set_title(
        f"Total Hardware Cost — MNIST MLP\n",
    )
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.45, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, (av, tv) in enumerate(zip(ann_tot_vals, tnn_tot_vals)):
        ax.text(i - width / 2, av * 1.08, f"{av:,}", ha="center", va="bottom", fontsize=8, rotation=15)
        ax.text(i + width / 2, tv * 1.08, f"{tv:,}", ha="center", va="bottom", fontsize=8, rotation=15)
    fig.tight_layout()
    _save_hw_fig(fig, "fig03_total_hardware_log_scale")


# --- Figure HW-5: TNN ternary weight distribution per layer ---
with plt.rc_context(_ACADEMIC_RC):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    n_pos = [L["n_pos"] for L in tnn_hw_layers]
    n_neg = [L["n_neg"] for L in tnn_hw_layers]
    n_zero = [L["n_zero"] for L in tnn_hw_layers]
    ax.bar(x, n_pos, width=0.55, label="$w = +1$", color="#2C5282", edgecolor="white")
    ax.bar(x, n_neg, width=0.55, bottom=n_pos, label="$w = -1$", color="#C05621", edgecolor="white")
    ax.bar(x, n_zero, width=0.55, bottom=np.array(n_pos) + np.array(n_neg),
           label="$w = 0$", color="#CBD5E0", edgecolor="white")
    ax.set_xticks(x, layer_labels)
    ax.set_ylabel("Weight count")
    ax.set_xlabel("Layer")
    ax.set_title("TNN Checkpoint: Ternary Weight Distribution per Layer")
    ax.legend(loc="upper right")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.45, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save_hw_fig(fig, "fig04_tnn_ternary_weight_stack")


# --- Figure HW-6: Synapse-level comparison table as figure ---
table_rows = []
for a, t in zip(ann_hw_layers, tnn_hw_layers):
    table_rows.append([
        a["name"],
        f"{a['out']}×{a['in']}",
        f"{a['synapses']:,}",
        f"{a['memristors']:,}",
        f"{t['memristors']:,}",
        f"{a['memristors'] / t['memristors']:.0f}×",
    ])
table_rows.append([
    "Total", "—",
    f"{ann_totals['synapses']:,}",
    f"{ann_totals['memristors']:,}",
    f"{tnn_totals['memristors']:,}",
    f"{total_ratio:.0f}×",
])

with plt.rc_context(_ACADEMIC_RC):
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.axis("off")
    col_labels = [
        "Layer", "Shape\n(out×in)", "Synapses",
        "ANN memristors", "TNN memristors", "Ratio",
    ]
    table = ax.table(
        cellText=table_rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    ax.set_title(
        "Hardware Mapping Summary (based on loaded fc1–fc3 weights)",
        pad=20, fontsize=12,
    )
    fig.tight_layout()
    _save_hw_fig(fig, "fig05_hardware_summary_table")







