from __future__ import annotations

'''
In The Name of God

Ali Pilehvar Meibody

Last Update : 07 sep 2026



09_Final_TNN_Dev_ADC/DAC_Paraisitic.py





In previous file (08) we see that effect of APM programmiogn error on TNN and also drift of apm
ON TNN . and now after full inevstigation of effect of device non-ideality.

it is time to consider also ACD/DAC and others also in this new file (09)



Here first we talk about ADC/DAC and what is that we go to optimization to select the dac , adc 
we just consider Quantizer for now, but for better (we can go to select the real like ramp and signed)
but for now even this need to consider mismatch and ... , so we can go to select only quantizer
between them we used th ehighest accuracy because of recovery . but in real case
we must trade-off with considering the cost and others ...



After that we can disable ADC/DAC (ideal) and only look at paraistic and different
and we said ok our final is 0.01 Ohm


so finally final one with enable Progrmaming, ADC/DAC and parisictic to see final one
but for better speed for next .py files we dont consider reisstance



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

import pandas as pd

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



TNN_APM_MODEL = "APM_SINW_SBFET"
SEED = 42
SEC_PER_DAY = 86400


DRIFT_TIMES_SEC = np.array([0, 10, 25, 50, 75, 100], dtype=float)



DRIFT_TIMES_DAYS = DRIFT_TIMES_SEC / SEC_PER_DAY



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




# ============================================================
# 4. Digital baseline

digital_tnn_acc = evaluate(net_tnn, test_loader, device="cpu")

print("Digital TNN accuracy:", digital_tnn_acc)
print("Convertible modules:")
print(convertible_modules(net_tnn))







    

################################################################################
################################################################################
################################################################################
################################################################################
'''                          TNN ADC/DAC level          '''
################################################################################
################################################################################
################################################################################
################################################################################



'''

Digital Input we have , because pixels or anything are saved in RAM in digital computer
so digital inputs comes, and it must go to DAC because we want to have Digital to analog converter
and then it go to crossbar MVM and we have phsyically MVM and finally each neuron which is one column
must output of that has soem activation and this activation can be go to ADC and pass to next layer

and again ......



Digital input --> DAC --> Crossbar mVM --> ADC --> Digital accumualtion --> Next layer



So DAC change inputs to analkog. Crossbar do matrix multiplciation and ADC output of analog
conevrt to digital.





-----------------------------------
DAC = digital to analog converter.
we have binary wiegthed DAC, R-2R ladder, sigma-delta, PWM and ....
so consider it used to generate input voltages, word-line signals, activation vectors and....

in crossSim :
    
    Quantizer DAC which is simplest mathematical DAC
        params.xbar.dac.mvm.model = "QuantizerDAC"
        which is ideal quantiZation only
        

    SignMagnitudeDAC which same as QuantizerDAC but support signed inputs using
        sign + magnitude representation.
    
    but also we have more real as we said like binary-weighted DAC, r-2r ladder,
        current-steering DAC, capacitive DAC (CDAS) 
    





-----------------------------------
ADC = Analog to Digital converter

the main role is one analog continious signals (like sensor voltage, voice, light
 temeprature) conveert to digital number to be processed.

we have resolution (bits) , sampling rate , and something...
In geenral we have Flash ADC (veery fast , high accuracy , high power consumption)
and used in radar and ... . the other is SAR ADC (sUCCESSIVE APPROXIMAITON REGISTER)
WHICH is most know in microcontroller has good accuracy, medium speed, low powe consumption
and it has applciations in Arduino , STM32, ... , also we have sigma-delta, dual slope,
pipeline ADC and ......

in cross sim specifically:
    QuantizerADC which is ideal quantizer only, no circuit behavior and good for
    clean circuit behavior and clean theoretical studies.
    
    
    SignMagnitudeADC which is same idea but signed output and useful with baalnce cores
    
    
    RampADC --> comapre input current agains a ramp signal over time. so it is 
    very simple , small area and energy efficient but slow
    
    
    SAR ADC --> most fmaous and work bianry search. excellent for energy efficiency
    good speed, but more ocmplicated and capcitor mismatch issues.
    
    Pipeline ADC --> very high speed, higher power , more analog compeity
    
    Cyclic ADC -> like iterative pipeline which is comapct an dmdoerante speed
    





------------------ Other considerations ------------------
So we have another thig , we can also do input bit slicing , for instance our
input is x =13 = 11012 and without slicign DAC most convert 13 to 0.65V and one time mVM
but with bit slciing, crossbar can do this 4 times for bit0,bit1,bit2,bit3 and finally summ

so here we have Tradeoff -------

With slicing                |        Without slicing
slow                        |        Fast
Tiny DAC                    |        Large DAC
More digital accumulation   |        More analog complexity
Higher latency              |        Lower latency
More robust                 |        Higher analog precision demand
More repeated MVM           |        More energy in DAC


so if we do this, because we have bti slicing, each bti has different MVM and
we must have ADC per ibit(True) . easier, relaistic,stabkle but higher nergy
higher latency 



Consider that DAC bits means that resolution input activation. when we have
dac_bits 8 means that our input has only 256 level inputs.
but we can start from .bits=0 and then go to see 2,4,6,8 and ... how much accuracy
degrade first for dac
and then for adc


IN ADC we have two error. one is quantization error. if range is high, space are higher
and clipping error we have means if signal go out of rnage we must clipped

so main challenge is here, if we have small range (quantization good, bad clipping)
ig we have big range(good clipping but bad quantizations) so we must go for calibrated
ADC.









    

cs_params_sliced.xbar.dac.mvm.model = 
cs_params_sliced.xbar.dac.vmm.model = 
cs_params_sliced.xbar.dac.mvm.bits = 
cs_params_sliced.xbar.dac.vmm.bits = 
cs_params_sliced.xbar.dac.mvm.input_bitslicing = 
cs_params_sliced.xbar.dac.vmm.input_bitslicing = 
cs_params_sliced.xbar.dac.mvm.slice_size =
cs_params_sliced.xbar.dac.vmm.slice_size = 

cs_params_sliced.xbar.adc.mvm.model = 
cs_params_sliced.xbar.adc.vmm.model = 
cs_params_sliced.xbar.adc.mvm.bits = 
cs_params_sliced.xbar.adc.vmm.bits = 
cs_params_sliced.xbar.adc.mvm.signed = 
cs_params_sliced.xbar.adc.vmm.signed = 
cs_params_sliced.xbar.adc.mvm.adc_per_ibit =
cs_params_sliced.xbar.adc.vmm.adc_per_ibit = 


'''






# ============================================================
# ADC/DAC decision sweep — no input bit slicing
# ============================================================
def build_tnn_apm_BASE_ADC_DAC_params(programming_enable=False, drift_enable=False, time=0.0):
    p = CrossSimParameters()

    # Array size
    p.core.rows_max = 1024
    p.core.cols_max = 1024

    # Balanced TNN core
    p.core.style = 1  # BALANCED
    p.core.balanced.style = 1  # ONE_SIDED
    p.core.balanced.interleaved_posneg = False
    p.core.balanced.subtract_current_in_xbar = True

    # No bit slicing for TNN
    p.core.weight_bits = 0
    p.core.bit_sliced.num_slices = 1

    # Device range
    p.xbar.device.Rmin = R_MIN
    p.xbar.device.Rmax = R_MAX
    p.xbar.device.cell_bits = 1
    p.xbar.device.Vread = 1.0

    # Time for drift model
    p.xbar.device.time = float(time)

    # APM device models
    p.xbar.device.programming_error.enable = programming_enable
    p.xbar.device.programming_error.model = TNN_APM_MODEL

    p.xbar.device.read_noise.enable = False

    p.xbar.device.drift_error.enable = drift_enable
    p.xbar.device.drift_error.model = TNN_APM_MODEL

    # Ideal ADC/DAC
    p.xbar.adc.mvm.bits = 0
    p.xbar.adc.vmm.bits = 0
    p.xbar.dac.mvm.bits = 0
    p.xbar.dac.vmm.bits = 0

    # No parasitics
    p.xbar.array.parasitics.enable = False
    p.xbar.array.parasitics.Rp_row = 0
    p.xbar.array.parasitics.Rp_col = 0
    p.xbar.array.parasitics.Rp_row_terminal = 0
    p.xbar.array.parasitics.Rp_col_terminal = 0
    p.xbar.array.parasitics.current_from_input = True
    p.xbar.array.parasitics.selected_rows = "top"

    return p



def run_peripheral_sweep(
    net_tnn,
    test_loader,
    adc_bits_list=(2, 3, 4),
    dac_bits_list=(4, 6),
    input_ranges=(4, 8, 16, 32),
    adc_ranges=(1, 2, 4),
    programming_enable=True,
    drift_enable=False,
    time=0.0,
    seed=42
):

    results = []

    # =========================================================
    # 0. IDEAL BASELINE
    # =========================================================

    p_ideal = build_tnn_apm_BASE_ADC_DAC_params(
        programming_enable=programming_enable,
        drift_enable=drift_enable,
        time=time
    )

    # Explicit ideal peripherals

    p_ideal.xbar.adc.mvm.model = "IdealADC"
    p_ideal.xbar.adc.vmm.model = "IdealADC"

    p_ideal.xbar.dac.mvm.model = "IdealDAC"
    p_ideal.xbar.dac.vmm.model = "IdealDAC"

    p_ideal.xbar.adc.mvm.bits = 0
    p_ideal.xbar.adc.vmm.bits = 0

    p_ideal.xbar.dac.mvm.bits = 0
    p_ideal.xbar.dac.vmm.bits = 0

    np.random.seed(seed)
    torch.manual_seed(seed)

    model_ideal = from_torch(net_tnn, p_ideal)
    model_ideal.eval()

    ideal_acc = evaluate(model_ideal, test_loader, device="cpu")

    print(f"\nIDEAL peripheral baseline accuracy = {ideal_acc:.2f}%\n")

    results.append({
        "mode": "ideal",
        "dac_bits": 0,
        "input_range": None,
        "adc_bits": 0,
        "adc_range": None,
        "accuracy": ideal_acc,
        "accuracy_drop": 0.0,
    })

    # =========================================================
    # 1. QUANTIZED SWEEP
    # =========================================================

    for dac_bits in dac_bits_list:
        for input_range in input_ranges:
            for adc_bits in adc_bits_list:
                for adc_range in adc_ranges:

                    p = build_tnn_apm_BASE_ADC_DAC_params(
                        programming_enable=programming_enable,
                        drift_enable=drift_enable,
                        time=time
                    )

                    # ---------------- DAC ----------------

                    p.xbar.dac.mvm.model = "QuantizerDAC"
                    p.xbar.dac.vmm.model = "QuantizerDAC"

                    p.xbar.dac.mvm.bits = dac_bits
                    p.xbar.dac.vmm.bits = dac_bits

                    p.xbar.dac.mvm.input_bitslicing = False
                    p.xbar.dac.vmm.input_bitslicing = False

                    p.core.mapping.inputs.mvm.min = -input_range
                    p.core.mapping.inputs.mvm.max = input_range

                    p.core.mapping.inputs.vmm.min = -input_range
                    p.core.mapping.inputs.vmm.max = input_range

                    # ---------------- ADC ----------------

                    p.xbar.adc.mvm.model = "QuantizerADC"
                    p.xbar.adc.vmm.model = "QuantizerADC"

                    p.xbar.adc.mvm.bits = adc_bits
                    p.xbar.adc.vmm.bits = adc_bits

                    p.xbar.adc.mvm.adc_range_option = 1
                    p.xbar.adc.vmm.adc_range_option = 1

                    p.xbar.adc.mvm.calibrated_range = [-adc_range, adc_range]
                    p.xbar.adc.vmm.calibrated_range = [-adc_range, adc_range]

                    np.random.seed(seed)
                    torch.manual_seed(seed)

                    model = from_torch(net_tnn, p)
                    model.eval()

                    acc = evaluate(model, test_loader, device="cpu")

                    acc_drop = ideal_acc - acc

                    results.append({
                        "mode": "quantized",
                        "dac_bits": dac_bits,
                        "input_range": input_range,
                        "adc_bits": adc_bits,
                        "adc_range": adc_range,
                        "accuracy": acc,
                        "accuracy_drop": acc_drop,
                    })

                    print(
                        f"DAC={dac_bits:>2} | "
                        f"Input ±{input_range:<5} | "
                        f"ADC={adc_bits:>2} | "
                        f"ADC range ±{adc_range:<4} | "
                        f"Acc={acc:.2f}% | "
                        f"Drop={acc_drop:.2f}%"
                    )

    return pd.DataFrame(results)




df_peripheral = run_peripheral_sweep(
    net_tnn=net_tnn,
    test_loader=test_loader
)


'''
26 may results

IDEAL peripheral baseline accuracy = 92.56%

DAC= 4 | Input ±4     | ADC= 2 | ADC range ±1    | Acc=10.81% | Drop=81.75%
DAC= 4 | Input ±4     | ADC= 2 | ADC range ±2    | Acc=14.84% | Drop=77.72%
DAC= 4 | Input ±4     | ADC= 2 | ADC range ±4    | Acc=20.77% | Drop=71.79%
DAC= 4 | Input ±4     | ADC= 3 | ADC range ±1    | Acc=11.27% | Drop=81.29%
DAC= 4 | Input ±4     | ADC= 3 | ADC range ±2    | Acc=18.67% | Drop=73.89%
DAC= 4 | Input ±4     | ADC= 3 | ADC range ±4    | Acc=28.08% | Drop=64.48%
DAC= 4 | Input ±4     | ADC= 4 | ADC range ±1    | Acc=11.43% | Drop=81.13%
DAC= 4 | Input ±4     | ADC= 4 | ADC range ±2    | Acc=24.45% | Drop=68.11%
DAC= 4 | Input ±4     | ADC= 4 | ADC range ±4    | Acc=43.67% | Drop=48.89%
DAC= 4 | Input ±8     | ADC= 2 | ADC range ±1    | Acc=12.21% | Drop=80.35%
DAC= 4 | Input ±8     | ADC= 2 | ADC range ±2    | Acc=14.86% | Drop=77.70%
DAC= 4 | Input ±8     | ADC= 2 | ADC range ±4    | Acc=22.29% | Drop=70.27%
DAC= 4 | Input ±8     | ADC= 3 | ADC range ±1    | Acc=12.26% | Drop=80.30%
DAC= 4 | Input ±8     | ADC= 3 | ADC range ±2    | Acc=22.75% | Drop=69.81%
DAC= 4 | Input ±8     | ADC= 3 | ADC range ±4    | Acc=28.56% | Drop=64.00%
DAC= 4 | Input ±8     | ADC= 4 | ADC range ±1    | Acc=12.52% | Drop=80.04%
DAC= 4 | Input ±8     | ADC= 4 | ADC range ±2    | Acc=28.44% | Drop=64.12%
DAC= 4 | Input ±8     | ADC= 4 | ADC range ±4    | Acc=49.42% | Drop=43.14%
DAC= 4 | Input ±16    | ADC= 2 | ADC range ±1    | Acc=12.14% | Drop=80.42%
DAC= 4 | Input ±16    | ADC= 2 | ADC range ±2    | Acc=12.76% | Drop=79.80%
DAC= 4 | Input ±16    | ADC= 2 | ADC range ±4    | Acc=18.92% | Drop=73.64%
DAC= 4 | Input ±16    | ADC= 3 | ADC range ±1    | Acc=12.10% | Drop=80.46%
DAC= 4 | Input ±16    | ADC= 3 | ADC range ±2    | Acc=21.37% | Drop=71.19%
DAC= 4 | Input ±16    | ADC= 3 | ADC range ±4    | Acc=24.69% | Drop=67.87%
DAC= 4 | Input ±16    | ADC= 4 | ADC range ±1    | Acc=12.44% | Drop=80.12%
DAC= 4 | Input ±16    | ADC= 4 | ADC range ±2    | Acc=24.58% | Drop=67.98%
DAC= 4 | Input ±16    | ADC= 4 | ADC range ±4    | Acc=41.86% | Drop=50.70%
DAC= 4 | Input ±32    | ADC= 2 | ADC range ±1    | Acc=10.53% | Drop=82.03%
DAC= 4 | Input ±32    | ADC= 2 | ADC range ±2    | Acc=10.63% | Drop=81.93%
DAC= 4 | Input ±32    | ADC= 2 | ADC range ±4    | Acc=13.93% | Drop=78.63%
DAC= 4 | Input ±32    | ADC= 3 | ADC range ±1    | Acc=10.75% | Drop=81.81%
DAC= 4 | Input ±32    | ADC= 3 | ADC range ±2    | Acc=16.04% | Drop=76.52%
DAC= 4 | Input ±32    | ADC= 3 | ADC range ±4    | Acc=16.92% | Drop=75.64%
DAC= 4 | Input ±32    | ADC= 4 | ADC range ±1    | Acc=11.17% | Drop=81.39%
DAC= 4 | Input ±32    | ADC= 4 | ADC range ±2    | Acc=15.34% | Drop=77.22%
DAC= 4 | Input ±32    | ADC= 4 | ADC range ±4    | Acc=24.96% | Drop=67.60%
DAC= 6 | Input ±4     | ADC= 2 | ADC range ±1    | Acc=39.05% | Drop=53.51%
DAC= 6 | Input ±4     | ADC= 2 | ADC range ±2    | Acc=26.59% | Drop=65.97%
DAC= 6 | Input ±4     | ADC= 2 | ADC range ±4    | Acc=27.41% | Drop=65.15%
DAC= 6 | Input ±4     | ADC= 3 | ADC range ±1    | Acc=59.96% | Drop=32.60%
DAC= 6 | Input ±4     | ADC= 3 | ADC range ±2    | Acc=58.44% | Drop=34.12%
DAC= 6 | Input ±4     | ADC= 3 | ADC range ±4    | Acc=36.37% | Drop=56.19%
DAC= 6 | Input ±4     | ADC= 4 | ADC range ±1    | Acc=59.12% | Drop=33.44%
DAC= 6 | Input ±4     | ADC= 4 | ADC range ±2    | Acc=68.65% | Drop=23.91%
DAC= 6 | Input ±4     | ADC= 4 | ADC range ±4    | Acc=61.03% | Drop=31.53%
DAC= 6 | Input ±8     | ADC= 2 | ADC range ±1    | Acc=39.08% | Drop=53.48%
DAC= 6 | Input ±8     | ADC= 2 | ADC range ±2    | Acc=27.04% | Drop=65.52%
DAC= 6 | Input ±8     | ADC= 2 | ADC range ±4    | Acc=27.81% | Drop=64.75%
DAC= 6 | Input ±8     | ADC= 3 | ADC range ±1    | Acc=68.23% | Drop=24.33%
DAC= 6 | Input ±8     | ADC= 3 | ADC range ±2    | Acc=57.84% | Drop=34.72%
DAC= 6 | Input ±8     | ADC= 3 | ADC range ±4    | Acc=36.45% | Drop=56.11%
DAC= 6 | Input ±8     | ADC= 4 | ADC range ±1    | Acc=73.06% | Drop=19.50%
DAC= 6 | Input ±8     | ADC= 4 | ADC range ±2    | Acc=73.12% | Drop=19.44%
DAC= 6 | Input ±8     | ADC= 4 | ADC range ±4    | Acc=60.53% | Drop=32.03%
DAC= 6 | Input ±16    | ADC= 2 | ADC range ±1    | Acc=38.06% | Drop=54.50%
DAC= 6 | Input ±16    | ADC= 2 | ADC range ±2    | Acc=26.81% | Drop=65.75%
DAC= 6 | Input ±16    | ADC= 2 | ADC range ±4    | Acc=26.69% | Drop=65.87%
DAC= 6 | Input ±16    | ADC= 3 | ADC range ±1    | Acc=69.83% | Drop=22.73%
DAC= 6 | Input ±16    | ADC= 3 | ADC range ±2    | Acc=56.40% | Drop=36.16%
DAC= 6 | Input ±16    | ADC= 3 | ADC range ±4    | Acc=34.44% | Drop=58.12%
DAC= 6 | Input ±16    | ADC= 4 | ADC range ±1    | Acc=75.26% | Drop=17.30%
DAC= 6 | Input ±16    | ADC= 4 | ADC range ±2    | Acc=74.28% | Drop=18.28%
DAC= 6 | Input ±16    | ADC= 4 | ADC range ±4    | Acc=59.50% | Drop=33.06%
DAC= 6 | Input ±32    | ADC= 2 | ADC range ±1    | Acc=35.85% | Drop=56.71%
DAC= 6 | Input ±32    | ADC= 2 | ADC range ±2    | Acc=25.10% | Drop=67.46%
DAC= 6 | Input ±32    | ADC= 2 | ADC range ±4    | Acc=26.14% | Drop=66.42%
DAC= 6 | Input ±32    | ADC= 3 | ADC range ±1    | Acc=69.27% | Drop=23.29%
DAC= 6 | Input ±32    | ADC= 3 | ADC range ±2    | Acc=50.47% | Drop=42.09%
DAC= 6 | Input ±32    | ADC= 3 | ADC range ±4    | Acc=33.74% | Drop=58.82%
DAC= 6 | Input ±32    | ADC= 4 | ADC range ±1    | Acc=81.14% | Drop=11.42%
DAC= 6 | Input ±32    | ADC= 4 | ADC range ±2    | Acc=74.65% | Drop=17.91%
DAC= 6 | Input ±32    | ADC= 4 | ADC range ±4    | Acc=55.29% | Drop=37.27%

'''










#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------

def plot_input_range_heatmaps(
    df_peripheral,
    fixed_dac_bits=6,
    metric="accuracy",
    use_quantized_only=True,
    dpi=600,
    save_path=None
):
    import numpy as np
    import matplotlib.pyplot as plt

    df = df_peripheral.copy()

    if use_quantized_only and "mode" in df.columns:
        df = df[df["mode"] == "quantized"].copy()

    df = df[df["dac_bits"] == fixed_dac_bits].copy()

    if df.empty:
        raise ValueError("No rows found for selected DAC bits.")

    input_ranges = sorted(df["input_range"].dropna().unique())
    ncols = len(input_ranges)

    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(4.15 * ncols, 4.35),
        dpi=dpi
    )

    if ncols == 1:
        axes = [axes]

    vmin = df[metric].min()
    vmax = df[metric].max()

    if metric == "accuracy_drop":
        cmap = "cividis_r"
        cbar_label = "Accuracy drop from ideal (%)"
        main_title = "Peripheral-Induced Accuracy Degradation"
        value_fmt = "{:.1f}"
        
        
    else:
        cmap = "cividis"
        cbar_label = "MNIST test accuracy (%)"
        main_title = "ADC/DAC Peripheral Optimization Landscape"
        value_fmt = "{:.1f}"

    im = None

    for ax, ir in zip(axes, input_ranges):
        dfi = df[df["input_range"] == ir].copy()

        pivot = dfi.pivot_table(
            index="adc_bits",
            columns="adc_range",
            values=metric,
            aggfunc="mean"
        ).sort_index().sort_index(axis=1)

        im = ax.imshow(
            pivot.values,
            aspect="equal",
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest"
        )

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"±{x:g}" for x in pivot.columns], fontsize=9)

        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"{int(x)}" for x in pivot.index], fontsize=9)

        ax.set_xlabel("ADC range", fontsize=10, labelpad=6)

        if ax is axes[0]:
            ax.set_ylabel("ADC resolution (bits)", fontsize=10, labelpad=6)
        else:
            ax.set_ylabel("")

        ax.set_title(
            f"Input range ±{ir:g}",
            fontsize=11,
            pad=9,
            fontweight="bold"
        )

        local_mid = 0.5 * (np.nanmin(pivot.values) + np.nanmax(pivot.values))

        
            
            
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                value = pivot.values[i, j]
                if not np.isnan(value):
                    
                    if metric == "accuracy_drop":
                        if value>local_mid :
                            font_colours = 'white'
                            
                        else:
                            font_colours = 'black'
                                    
                    else:
                        if value>local_mid :
                            font_colours = 'black'
                            
                        else:
                            font_colours = 'white'
                    ax.text(
                        j,
                        i,
                        value_fmt.format(value),
                        ha="center",
                        va="center",
                        fontsize=8.2,
                        fontweight="bold",
                        color=font_colours
                    )

        ax.tick_params(axis="both", length=3, width=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.subplots_adjust(
        left=0.055,
        right=0.905,
        bottom=0.16,
        top=0.78,
        wspace=0.24
    )

    cax = fig.add_axes([0.925, 0.22, 0.014, 0.47])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(cbar_label, fontsize=10, labelpad=9)
    cbar.ax.tick_params(labelsize=8.5, length=3)

    fig.suptitle(
        f"{main_title}\nDAC resolution = {fixed_dac_bits}-bit",
        fontsize=14,
        fontweight="bold",
        y=0.93
    )

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()
    
    


#also accuracy_drop
plot_input_range_heatmaps(
    df_peripheral,
    fixed_dac_bits=4,
    metric="accuracy"
)


  
plot_input_range_heatmaps(
    df_peripheral,
    fixed_dac_bits=6,
    metric="accuracy"
)



















#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------


def plot_ideal_vs_selected_configs_academic(
    df_peripheral,
    selected_configs,
    dpi=600,
    save_path_prefix=None
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # ============================================================
    # Academic style
    # ============================================================
    COLOR_IDEAL = "#1F3B73"
    COLOR_CFG1  = "#4C72B0"
    COLOR_CFG2  = "#C17C2F"
    COLOR_CFG3  = "#8C5A2B"
    COLORS = [COLOR_IDEAL, COLOR_CFG1, COLOR_CFG2, COLOR_CFG3]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    df = df_peripheral.copy()

    if "mode" in df.columns:
        ideal_rows = df[df["mode"] == "ideal"].copy()
        dfq = df[df["mode"] == "quantized"].copy()
    else:
        ideal_rows = df[df["adc_bits"] == 0].copy()
        dfq = df[df["adc_bits"] > 0].copy()

    ideal_acc = ideal_rows["accuracy"].iloc[0] if not ideal_rows.empty else df["accuracy"].max()

    # ============================================================
    # Select configs
    # ============================================================
    rows = []

    for cfg in selected_configs:
        mask = (
            (dfq["dac_bits"] == cfg["dac_bits"]) &
            (dfq["adc_bits"] == cfg["adc_bits"]) &
            (dfq["input_range"] == cfg["input_range"]) &
            (dfq["adc_range"] == cfg["adc_range"])
        )

        hit = dfq[mask]

        if hit.empty:
            raise ValueError(f"Config not found: {cfg}")

        rows.append(hit.iloc[0])

    selected = pd.DataFrame(rows).reset_index(drop=True)

    labels = ["Ideal\nADC/DAC"]
    accs = [float(ideal_acc)]

    for _, r in selected.iterrows():
        labels.append(
            f"DAC {int(r['dac_bits'])}-bit\n"
            f"ADC {int(r['adc_bits'])}-bit\n"
            f"In ±{r['input_range']:g}\n"
            f"ADC ±{r['adc_range']:g}"
        )
        accs.append(float(r["accuracy"]))

    drops = [0.0] + [ideal_acc - a for a in accs[1:]]
    x = np.arange(len(labels))

    # ============================================================
    # Approximate hardware-energy proxy for selected configs only
    # Network: 784 → 256 → 128 → 10
    # ============================================================
    layer_shapes = [
        (784, 256),
        (256, 128),
        (128, 10)
    ]

    total_output_neurons = sum(o for _, o in layer_shapes)
    total_input_drives = sum(i for i, _ in layer_shapes)
    total_macs = sum(i * o for i, o in layer_shapes)

    selected = selected.copy()

    selected["adc_energy_proxy"] = (
        total_output_neurons * (2 ** selected["adc_bits"])
    )

    selected["dac_energy_proxy"] = (
        0.35 * total_input_drives * (2 ** selected["dac_bits"])
    )

    selected["xbar_compute_proxy"] = (
        0.015 * total_macs
    )


    selected["range_cost_proxy"] = np.log2(selected["input_range"])
    selected["total_energy_proxy"] = (
        selected["adc_energy_proxy"]
        + selected["dac_energy_proxy"]
        + selected["xbar_compute_proxy"]
        + 500 * selected["range_cost_proxy"]
    )

    selected["accuracy_per_energy"] = (
        selected["accuracy"] / selected["total_energy_proxy"]
    )

    selected["relative_efficiency"] = (
        selected["accuracy_per_energy"] /
        selected["accuracy_per_energy"].max()
    )
    
    selected["relative_peripheral_cost"] = (
            (2 ** selected["adc_bits"]) *
            np.log2(selected["input_range"] + 1) *
            np.log2(selected["adc_range"] + 1)
        )

    # ============================================================
    # FIGURE 1 — Accuracy comparison
    # ============================================================
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=dpi)

    bars = ax.bar(
        x,
        accs,
        width=0.58,
        color=COLORS[:len(accs)],
        edgecolor="black",
        linewidth=0.9,
        zorder=3
    )

    ax.set_ylabel("MNIST Test Accuracy (%)", fontsize=11)
    ax.set_title(
        "Effect of Low-Precision Peripheral Quantization on TNN Inference Accuracy",
        fontsize=13,
        fontweight="bold",
        pad=12
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.7)
    ax.set_ylim(max(0, min(accs) - 8), 100)

    ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.25, zorder=0)
    ax.set_axisbelow(True)

    for bar, val in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.55,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path_prefix is not None:
        plt.savefig(
            f"{save_path_prefix}_accuracy_comparison.png",
            dpi=dpi,
            bbox_inches="tight"
        )

    plt.show()

    # ============================================================
    # FIGURE 2 — Accuracy degradation
    # ============================================================
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=dpi)

    for i, (drop, color) in enumerate(zip(drops, COLORS[:len(drops)])):
        ax.plot(
            [i, i],
            [0, drop],
            color=color,
            linewidth=3,
            solid_capstyle="round",
            zorder=2
        )

        ax.scatter(
            i,
            drop,
            s=135,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=3
        )

        ax.text(
            i,
            drop + 0.45,
            f"{drop:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.7)
    ax.set_ylabel("Accuracy Drop from Ideal (%)", fontsize=11)

    ax.set_title(
        "Peripheral Quantization-Induced Accuracy Loss",
        fontsize=13,
        fontweight="bold",
        pad=12
    )

    ax.set_ylim(0, max(drops) + 5)
    ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.25)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path_prefix is not None:
        plt.savefig(
            f"{save_path_prefix}_accuracy_drop.png",
            dpi=dpi,
            bbox_inches="tight"
        )

    plt.show()

    # ============================================================
    # FIGURE 3 — Clean Accuracy vs Peripheral Burden Trade-off
    # ============================================================
    
    fig, ax1 = plt.subplots(figsize=(7.6, 4.8), dpi=dpi)
    
    config_names = [
        f"Input ±{r['input_range']:g}"
        for _, r in selected.iterrows()
    ]
    
    x_cfg = np.arange(len(selected))
    
    # Normalize burden to the lowest-burden config
    selected["relative_burden_norm"] = (
        selected["relative_peripheral_cost"] /
        selected["relative_peripheral_cost"].min()
    )
    
    # Accuracy line
    ax1.plot(
        x_cfg,
        selected["accuracy"],
        marker="o",
        linewidth=2.2,
        markersize=7,
        color=COLOR_IDEAL,
        label="Accuracy"
    )
    
    ax1.set_ylabel("MNIST Test Accuracy (%)", fontsize=11)
    ax1.set_ylim(
        selected["accuracy"].min() - 4,
        min(100, selected["accuracy"].max() + 8)
    )
    
    ax1.axhline(
        ideal_acc,
        linestyle="--",
        linewidth=1.1,
        color=COLOR_IDEAL,
        alpha=0.75   )
    
    for i, acc in enumerate(selected["accuracy"]):
        ax1.text(
            i,
            acc + 0.7,
            f"{acc:.2f}%",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color=COLOR_IDEAL
        )
    
    # Burden bar on second axis
    ax2 = ax1.twinx()
    
    bars = ax2.bar(
        x_cfg,
        selected["relative_burden_norm"],
        width=0.45,
        alpha=0.28,
        color="#C17C2F",
        edgecolor="black",
        linewidth=0.8,
        label="Relative peripheral burden"
    )
    
    ax2.set_ylabel("Relative Peripheral Burden (normalized)", fontsize=11)
    ax2.set_ylim(0, selected["relative_burden_norm"].max() * 1.35)
    
    for bar, val in zip(bars, selected["relative_burden_norm"]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.05,
            f"{val:.2f}×",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#8C5A2B"
        )
    
    ax1.set_xticks(x_cfg)
    ax1.set_xticklabels(config_names, fontsize=10)
    
    ax1.set_xlabel("Selected Input Dynamic Range", fontsize=11)
    
    ax1.set_title(
        "Accuracy–Peripheral Burden Trade-off for Selected Configurations",
        fontsize=13,
        fontweight="bold",
        pad=12
    )
    
    ax1.grid(
        axis="y",
        linestyle="--",
        linewidth=0.55,
        alpha=0.25
    )
    
    # Cleaner legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        frameon=False,
        fontsize=9,
        loc="upper left"
    )
    
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    
    plt.tight_layout()
    
    if save_path_prefix is not None:
        plt.savefig(
            f"{save_path_prefix}_accuracy_burden_tradeoff_clean.png",
            dpi=dpi,
            bbox_inches="tight"
        )
    
    plt.show()

    # ============================================================
    # Summary
    # ============================================================
    summary = selected[[
        "dac_bits",
        "adc_bits",
        "input_range",
        "adc_range",
        "accuracy",
        "adc_energy_proxy",
        "dac_energy_proxy",
        "xbar_compute_proxy",
        "total_energy_proxy",
        "accuracy_per_energy",
        "relative_efficiency"
    ]].copy()

    summary["drop_vs_ideal"] = ideal_acc - summary["accuracy"]

    print("\nSelected configurations with estimated hardware-efficiency proxy:")
    print(summary)

    return summary






summary = plot_ideal_vs_selected_configs_academic(
    df_peripheral,
    selected_configs=[
        {"dac_bits": 6, "adc_bits": 4, "input_range": 8,  "adc_range": 1},
        {"dac_bits": 6, "adc_bits": 4, "input_range": 16, "adc_range": 1},
        {"dac_bits": 6, "adc_bits": 4, "input_range": 32, "adc_range": 1},
    ],
    dpi=600)




#for dffinding best
best_config = df_peripheral.sort_values("accuracy", ascending=False).iloc[0]

print("Best peripheral configuration:")
print(best_config)









# ============================================================
# REAL ADC/DAC only for comparisons
# ============================================================

DAC_BITS =6
ADC_BITS=4
INPUT_RANGE=32
ADC_RANGE=1




def build_tnn_apm_params(
    programming_enable=False,
    drift_enable=False,
    time=0.0,
    peripheral_mode="ideal"
):
    p = CrossSimParameters()

    # Core / mapping

    p.core.rows_max = 1024
    p.core.cols_max = 1024

    # Balanced core for ternary weights: -1, 0, +1
    p.core.style = 1
    p.core.balanced.style = 1
    p.core.balanced.interleaved_posneg = False
    p.core.balanced.subtract_current_in_xbar = True

    # No weight bit slicing for TNN
    p.core.weight_bits = 0
    p.core.bit_sliced.num_slices = 1
    
    INPUT_CALIBRATION_RANGE = INPUT_RANGE
    ADC_CALIBRATION_RANGE = ADC_RANGE



    # Symmetric input range is important for SignMagnitudeDAC
    p.core.mapping.inputs.mvm.min = -1*INPUT_CALIBRATION_RANGE
    p.core.mapping.inputs.mvm.max = INPUT_CALIBRATION_RANGE
    p.core.mapping.inputs.vmm.min = -1*INPUT_CALIBRATION_RANGE
    p.core.mapping.inputs.vmm.max = INPUT_CALIBRATION_RANGE

    # APM device

    p.xbar.device.Rmin = R_MIN  # LRS / ON
    p.xbar.device.Rmax = R_MAX   # HRS / OFF
    p.xbar.device.cell_bits = 1
    p.xbar.device.Vread = 1.0
    p.xbar.device.time = float(time)

    p.xbar.device.programming_error.enable = programming_enable
    p.xbar.device.programming_error.model = TNN_APM_MODEL

    p.xbar.device.read_noise.enable = False

    p.xbar.device.drift_error.enable = drift_enable
    p.xbar.device.drift_error.model = TNN_APM_MODEL

    # ADC / DAC modes

    if peripheral_mode == "ideal":
        # No ADC/DAC quantization
        p.xbar.adc.mvm.model = "IdealADC"
        p.xbar.adc.vmm.model = "IdealADC"
        p.xbar.dac.mvm.model = "IdealDAC"
        p.xbar.dac.vmm.model = "IdealDAC"

        p.xbar.adc.mvm.bits = 0
        p.xbar.adc.vmm.bits = 0
        p.xbar.dac.mvm.bits = 0
        p.xbar.dac.vmm.bits = 0


    elif peripheral_mode == "quantizer":
        # DAC: ideal mathematical quantization
    
        p.xbar.dac.mvm.model = "QuantizerDAC"
        p.xbar.dac.vmm.model = "QuantizerDAC"
    
        p.xbar.dac.mvm.bits = DAC_BITS
        p.xbar.dac.vmm.bits = DAC_BITS
    
        #p.xbar.dac.mvm.signed = True
        #p.xbar.dac.vmm.signed = True
    
        p.xbar.dac.mvm.input_bitslicing = False
        p.xbar.dac.vmm.input_bitslicing = False
    
    
        # ADC: ideal mathematical quantization
    
        p.xbar.adc.mvm.model = "QuantizerADC"
        p.xbar.adc.vmm.model = "QuantizerADC"
    
        p.xbar.adc.mvm.bits = ADC_BITS
        p.xbar.adc.vmm.bits = ADC_BITS
    
        #p.xbar.adc.mvm.signed = True
        #p.xbar.adc.vmm.signed = True
    
        p.xbar.adc.mvm.adc_per_ibit = False
        p.xbar.adc.vmm.adc_per_ibit = False
    
    
        # CRITICAL:
        # use calibrated range, not MAX/GRANULAR
    
        p.xbar.adc.mvm.adc_range_option = 1
        p.xbar.adc.vmm.adc_range_option = 1
    
        p.xbar.adc.mvm.calibrated_range = [-1*ADC_CALIBRATION_RANGE, ADC_CALIBRATION_RANGE]
        p.xbar.adc.vmm.calibrated_range = [-1*ADC_CALIBRATION_RANGE, ADC_CALIBRATION_RANGE]
        

    
    elif peripheral_mode == "realistic":
        # DAC
        
        p.xbar.dac.mvm.model = "SignMagnitudeDAC"
        p.xbar.dac.vmm.model = "SignMagnitudeDAC"
    
        p.xbar.dac.mvm.bits = DAC_BITS
        p.xbar.dac.vmm.bits = DAC_BITS
        
        p.xbar.dac.mvm.signed = True
        p.xbar.dac.vmm.signed = True
        
        p.xbar.dac.mvm.input_bitslicing = False
        p.xbar.dac.vmm.input_bitslicing = False
        
        
        # ADC
        
        p.xbar.adc.mvm.model = "SarADC"
        p.xbar.adc.vmm.model = "SarADC"
        
        p.xbar.adc.mvm.bits = ADC_BITS
        p.xbar.adc.vmm.bits = ADC_BITS
        
        p.xbar.adc.mvm.signed = True
        p.xbar.adc.vmm.signed = True
        
        p.xbar.adc.mvm.adc_per_ibit = False
        p.xbar.adc.vmm.adc_per_ibit = False
        
        
        # CRITICAL FIX:
        # calibrated ADC range
        
        p.xbar.adc.mvm.adc_range_option = 1
        p.xbar.adc.vmm.adc_range_option = 1
        
        p.xbar.adc.mvm.calibrated_range = [-1*ADC_CALIBRATION_RANGE,ADC_CALIBRATION_RANGE]
        p.xbar.adc.vmm.calibrated_range = [-1*ADC_CALIBRATION_RANGE,ADC_CALIBRATION_RANGE]
        
        
        # SAR nonidealities
        
        p.xbar.adc.mvm.gain_db = 100
        p.xbar.adc.vmm.gain_db = 100
        
        p.xbar.adc.mvm.sigma_capacitor = 0.0
        p.xbar.adc.vmm.sigma_capacitor = 0.0
        
        p.xbar.adc.mvm.sigma_comparator = 0.0
        p.xbar.adc.vmm.sigma_comparator = 0.0
        
        p.xbar.adc.mvm.split_cdac = True
        p.xbar.adc.vmm.split_cdac = True
        
        p.xbar.adc.mvm.group_size = 128
        p.xbar.adc.vmm.group_size = 128

    else:
        raise ValueError(
            "peripheral_mode must be: ideal, quantizer, or realistic"
        )

    # Parasitics
    # Keep disabled for ADC/DAC comparison

    p.xbar.array.parasitics.enable = False
    p.xbar.array.parasitics.Rp_row = 0
    p.xbar.array.parasitics.Rp_col = 0
    p.xbar.array.parasitics.Rp_row_terminal = 0
    p.xbar.array.parasitics.Rp_col_terminal = 0
    p.xbar.array.parasitics.current_from_input = True
    p.xbar.array.parasitics.selected_rows = "top"
    return p









# ============================================================
# 3. Compare TNN peripheral scenarios + store models/predictions

torch.manual_seed(SEED)
np.random.seed(SEED)

def get_predictions(model, test_loader, device="cpu", max_batches=None):
    model.eval()
    all_y = []
    all_pred = []
    all_logits = []

    with torch.no_grad():
        for b, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            pred = out.argmax(dim=1)

            all_y.append(y.cpu())
            all_pred.append(pred.cpu())
            all_logits.append(out.cpu())

            if max_batches is not None and b + 1 >= max_batches:
                break

    return (
        torch.cat(all_y).numpy(),
        torch.cat(all_pred).numpy(),
        torch.cat(all_logits).numpy()
    )


digital_tnn_acc = evaluate(net_tnn, test_loader, device="cpu")
y_true, pred_digital, logits_digital = get_predictions(net_tnn, test_loader)

# Match build_tnn_apm_params: QuantizerDAC/ADC and SignMagnitudeDAC/SarADC bit widths
TNN_PERIPH_DAC_BITS = DAC_BITS
TNN_PERIPH_ADC_BITS = ADC_BITS

_SCEN_BASELINE = "Baseline\nNo ADC/DAC"
_SCEN_QUANTIZER = f"Quantizer\n{TNN_PERIPH_DAC_BITS}b DAC / {TNN_PERIPH_ADC_BITS}b ADC"
_SCEN_REALISTIC = f"SignMagDAC\n{TNN_PERIPH_DAC_BITS}b / SAR {TNN_PERIPH_ADC_BITS}b"

scenarios = {
    _SCEN_BASELINE: "ideal",
    _SCEN_QUANTIZER: "quantizer",
    _SCEN_REALISTIC: "realistic",
}

results = []
models = {}
predictions = {}
logits_all = {}

for scenario_name, mode in scenarios.items():
    
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print("\nRunning:", scenario_name)

    params = build_tnn_apm_params(
        programming_enable=True,
        drift_enable=False,
        time=0.0,
        peripheral_mode=mode
    )

    analog_model = from_torch(net_tnn, params)
    analog_model.eval()

    y_true_tmp, pred_tmp, logits_tmp = get_predictions(analog_model, test_loader)
    acc = 100 * np.mean(pred_tmp == y_true_tmp)

    models[scenario_name] = analog_model
    predictions[scenario_name] = pred_tmp
    logits_all[scenario_name] = logits_tmp

    results.append({
        "Scenario": scenario_name,
        "Accuracy (%)": acc,
        "Drop vs Digital (%)": digital_tnn_acc - acc,
        "Mismatch vs Digital (%)": 100 * np.mean(pred_tmp != pred_digital),
    })

results_df = pd.DataFrame(results)

print("\n================ TNN ADC/DAC COMPARISON ================")
print("Digital TNN accuracy:", digital_tnn_acc)
print(results_df)


'''
Running: Baseline
No ADC/DAC

Running: Quantizer
6b DAC / 4b ADC

Running: SignMagDAC
6b / SAR 4b

================ TNN ADC/DAC COMPARISON ================
Digital TNN accuracy: 93.78
                     Scenario  ...  Mismatch vs Digital (%)
0        Baseline\nNo ADC/DAC  ...                     6.12
1  Quantizer\n6b DAC / 4b ADC  ...                    19.61
2     SignMagDAC\n6b / SAR 4b  ...                    27.28

[3 rows x 4 columns]

'''




# ============================================================
# 4–7. Academic figures — TNN ADC/DAC peripheral comparison

try:
    _PERIPH_RC = _ACADEMIC_RC
except NameError:
    _PERIPH_RC = {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "axes.titleweight": "normal",
        "axes.edgecolor": "#333333",
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#CCCCCC",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }

try:
    _periph_fig_dir = PROJECT_DIR / "figures" / "cross_sim2_tnn_adc_dac"
except NameError:
    _periph_fig_dir = Path.cwd() / "figures" / "cross_sim2_tnn_adc_dac"
_periph_fig_dir.mkdir(parents=True, exist_ok=True)
_PERIPH_DPI = 150

_COLOR_PERIPH_DIGITAL = "#2C5282"
_COLOR_PERIPH_IDEAL = "#718096"
_COLOR_PERIPH_QUANT = "#3182CE"
_COLOR_PERIPH_REAL = "#C05621"

_PERIPH_SCENARIO_COLORS = {
    _SCEN_BASELINE: _COLOR_PERIPH_IDEAL,
    _SCEN_QUANTIZER: _COLOR_PERIPH_QUANT,
    _SCEN_REALISTIC: _COLOR_PERIPH_REAL,
}
_PERIPH_SCENARIO_SHORT = {
    _SCEN_BASELINE: "Ideal\n(no ADC/DAC)",
    _SCEN_QUANTIZER: f"Quantizer\n{TNN_PERIPH_DAC_BITS}b DAC / {TNN_PERIPH_ADC_BITS}b ADC",
    _SCEN_REALISTIC: f"SignMagDAC {TNN_PERIPH_DAC_BITS}b\n+ SAR ADC {TNN_PERIPH_ADC_BITS}b",
}


def _save_periph_fig(fig, stem: str) -> None:
    #fig.savefig(_periph_fig_dir / f"{stem}.pdf", bbox_inches="tight", dpi=300)
    #fig.savefig(_periph_fig_dir / f"{stem}.png", bbox_inches="tight", dpi=_PERIPH_DPI)
    plt.show()


def _style_periph_ax(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.45, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _bar_colors_for_df(df) -> list[str]:
    return [_PERIPH_SCENARIO_COLORS.get(s, "#4A5568") for s in df["Scenario"]]


def _short_xtick_labels(df) -> list[str]:
    return [_PERIPH_SCENARIO_SHORT.get(s, s.replace("\n", " ")) for s in df["Scenario"]]


_scenario_names = list(results_df["Scenario"])
_bar_colors = _bar_colors_for_df(results_df)
_xtick_labels = _short_xtick_labels(results_df)
_x_pos = np.arange(len(_scenario_names))


# --- Figure 4a: Accuracy ---
_accs = np.asarray(results_df["Accuracy (%)"], dtype=float)
_digital = float(digital_tnn_acc)

with plt.rc_context(_PERIPH_RC):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(
        _x_pos, _accs, color=_bar_colors, width=0.58,
        edgecolor="white", linewidth=0.9, zorder=3,
    )
    ax.axhline(
        _digital, color=_COLOR_PERIPH_DIGITAL, linestyle="--", linewidth=1.4,
        label=f"Digital TNN ({_digital:.2f}%)", zorder=2,
    )
    y_min = min(_accs.min(), _digital) - 0.8
    y_max = max(_accs.max(), _digital) + 1.2
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(_x_pos)
    ax.set_xticklabels(_xtick_labels)
    ax.set_ylabel("MNIST test accuracy (%)")
    ax.set_title(
        "TNN inference accuracy under ADC/DAC peripheral models\n"
        f"({TNN_APM_MODEL}, {TNN_PERIPH_DAC_BITS}-bit DAC / {TNN_PERIPH_ADC_BITS}-bit ADC, "
        "programming & drift on, parasitics off)",
    )
    for bar, acc in zip(bars, _accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, acc + 0.06,
            f"{acc:.2f}%", ha="center", va="bottom", fontsize=9.5, color="#333333",
        )
    _style_periph_ax(ax)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    _save_periph_fig(fig, "fig01_accuracy_adc_dac")


# --- Figure 4b: Drop vs digital ---
with plt.rc_context(_PERIPH_RC):
    fig, ax = plt.subplots(figsize=(8.2, 5))
    drops = results_df["Drop vs Digital (%)"].values
    bars = ax.bar(
        _x_pos, drops, color=_bar_colors, width=0.58,
        edgecolor="white", linewidth=0.9, zorder=3,
    )
    ax.set_xticks(_x_pos, _xtick_labels)
    ax.set_ylabel("Accuracy drop vs digital TNN (pp)")
    ax.set_title("Accuracy loss induced by ADC/DAC quantization")
    for bar, drop in zip(bars, drops):
        ax.text(
            bar.get_x() + bar.get_width() / 2, drop + 0.05,
            f"{drop:.2f}", ha="center", va="bottom", fontsize=9.5, color="#333333",
        )
    _style_periph_ax(ax)
    fig.tight_layout()
    _save_periph_fig(fig, "fig02_accuracy_drop_adc_dac")


# --- Figure 5: Prediction mismatch vs digital ---
with plt.rc_context(_PERIPH_RC):
    fig, ax = plt.subplots(figsize=(8.2, 5))
    mismatch = results_df["Mismatch vs Digital (%)"].values
    bars = ax.bar(
        _x_pos, mismatch, color=_bar_colors, width=0.58,
        edgecolor="white", linewidth=0.9, zorder=3,
    )
    ax.set_xticks(_x_pos, _xtick_labels)
    ax.set_ylabel("Prediction mismatch vs digital (%)")
    ax.set_title("Fraction of test images with changed class prediction")
    for bar, val in zip(bars, mismatch):
        ax.text(
            bar.get_x() + bar.get_width() / 2, val + 0.15,
            f"{val:.2f}%", ha="center", va="bottom", fontsize=9.5, color="#333333",
        )
    _style_periph_ax(ax)
    fig.tight_layout()
    _save_periph_fig(fig, "fig03_prediction_mismatch_adc_dac")


# ============================================================
# 6. Confusion matrices (academic colormap)

def confusion_matrix_np(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix_academic(cm, title: str, stem: str) -> None:
    with plt.rc_context(_PERIPH_RC):
        fig, ax = plt.subplots(figsize=(6.4, 5.6))
        vmax = cm.max() if cm.max() > 0 else 1
        im = ax.imshow(cm, aspect="auto", cmap="Blues", vmin=0, vmax=vmax)
        ax.set_title(title, pad=10)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks(np.arange(10))
        ax.set_yticks(np.arange(10))
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Count", rotation=270, labelpad=12)
        thresh = vmax / 2.0
        for i in range(10):
            for j in range(10):
                val = cm[i, j]
                if val > 0:
                    ax.text(
                        j, i, str(val), ha="center", va="center", fontsize=7,
                        color="white" if val > thresh else "#1A202C",
                    )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        _save_periph_fig(fig, stem)


cm_digital = confusion_matrix_np(y_true, pred_digital)
plot_confusion_matrix_academic(
    cm_digital, "Digital TNN — confusion matrix", "fig04_cm_digital",
)

for scenario_name in scenarios.keys():
    cm = confusion_matrix_np(y_true, predictions[scenario_name])
    short = _PERIPH_SCENARIO_SHORT.get(scenario_name, scenario_name).replace("\n", " ")
    safe = scenario_name.replace("\n", "_").replace(" ", "_").replace("/", "")
    plot_confusion_matrix_academic(
        cm, f"{short} — confusion matrix", f"fig04_cm_{safe}",
    )










################################################################################
################################################################################
################################################################################
################################################################################
'''                            Paraisitics Errors                '''
################################################################################
################################################################################
################################################################################
################################################################################

#"quantizer"

parasitic_cases = {
    "Ideal\nRp=0": 0.0,
    "Rp=0.01Ω": 0.01,
    "Rp=0.1Ω": 0.1,
    "Rp=1Ω": 1.0,
    "Rp=10Ω": 10.0
}

parasitic_results = []

for case_name, Rp in parasitic_cases.items():
    
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    print("\nRunning parasitic case:", case_name)

    params = build_tnn_apm_params(
        programming_enable=True,
        drift_enable=True,
        time=0.0,
        peripheral_mode="ideal"
    )

    # Ideal ADC/DAC
    params.xbar.adc.mvm.bits = 0
    params.xbar.adc.vmm.bits = 0
    params.xbar.dac.mvm.bits = 0
    params.xbar.dac.vmm.bits = 0

    # Only parasitics change
    params.xbar.array.parasitics.enable = Rp > 0
    params.xbar.array.parasitics.Rp_row = Rp
    params.xbar.array.parasitics.Rp_col = Rp
    params.xbar.array.parasitics.current_from_input = True
    params.xbar.array.parasitics.selected_rows = "top"

    analog_model = from_torch(net_tnn, params)
    analog_model.eval()

    acc = evaluate(analog_model, test_loader, device="cpu")

    parasitic_results.append({
        "Case": case_name,
        "Rp_ohm": Rp,
        "Accuracy (%)": acc,
    })

parasitic_df = pd.DataFrame(parasitic_results)

baseline_acc = parasitic_df.iloc[0]["Accuracy (%)"]
parasitic_df["Drop vs Ideal (%)"] = baseline_acc - parasitic_df["Accuracy (%)"]

print("\n================ PARASITICS ONLY RESULTS ================")
print(parasitic_df)


'''

Running parasitic case: Ideal
Rp=0

Running parasitic case: Rp=0.01Ω

Running parasitic case: Rp=0.1Ω

Running parasitic case: Rp=1Ω

Running parasitic case: Rp=10Ω

================ PARASITICS ONLY RESULTS ================
          Case  Rp_ohm  Accuracy (%)  Drop vs Ideal (%)
0  Ideal\nRp=0    0.00         92.56                0.0
1     Rp=0.01Ω    0.01         92.56                0.0
2      Rp=0.1Ω    0.10         92.56                0.0
3        Rp=1Ω    1.00         92.56                0.0
4       Rp=10Ω   10.00         92.56                0.0





R_MIN = 1.410360e+09 Ω
R_MAX = 4.429246e+09 Ω


I_ON = Vread / R_MIN
     = 1 / 1.410360e9
     ≈ 0.709 nA
     
    


Worst-case column current with 1024 active cells:

I_col ≈ 1024 × 0.709 nA
      ≈ 726 nA
      
      
even with 10 ohm
V_drop = I_col × Rp
       ≈ 726 nA × 10 Ω
       ≈ 7.26 µV
       


compaed to 1 V read voltage
V_drop / Vread ≈ 0.000726%

So parasitic voltage drop is basically negligible. That is why all accuracies stay exactly 92.56% after you fixed the seed.



'''



VREAD = 1.0
RMIN = R_MIN
I_cell_max = VREAD / RMIN
N_active = 1024
parasitic_df["I_worst_col_A"] = N_active * I_cell_max
parasitic_df["Estimated_Vdrop_V"] = parasitic_df["I_worst_col_A"] * parasitic_df["Rp_ohm"]
parasitic_df["Estimated_Vdrop_%Vread"] = 100 * parasitic_df["Estimated_Vdrop_V"] / VREAD
print("\n================ ESTIMATED VOLTAGE DROP ================")
print(parasitic_df[["Case", "Rp_ohm", "Estimated_Vdrop_V", "Estimated_Vdrop_%Vread"]])

'''
================ ESTIMATED VOLTAGE DROP ================
          Case  Rp_ohm  Estimated_Vdrop_V  Estimated_Vdrop_%Vread
0  Ideal\nRp=0    0.00       0.000000e+00            0.000000e+00
1     Rp=0.01Ω    0.01       7.260558e-09            7.260558e-07
2      Rp=0.1Ω    0.10       7.260558e-08            7.260558e-06
3        Rp=1Ω    1.00       7.260558e-07            7.260558e-05
4       Rp=10Ω   10.00       7.260558e-06            7.260558e-04

'''

try:
    _PARASITIC_RC = _PERIPH_RC
except NameError:
    _PARASITIC_RC = {"font.family": "serif", "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13}

def _save_parasitic_fig(fig, stem: str) -> None:
    #fig.savefig(_parasitic_fig_dir / f"{stem}.pdf", bbox_inches="tight", dpi=300)
    #fig.savefig(_parasitic_fig_dir / f"{stem}.png", bbox_inches="tight", dpi=150)
    plt.show()


def _style_parasitic_ax(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, which="major", alpha=0.45, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


_rp = parasitic_df["Rp_ohm"].to_numpy(dtype=float)
_accs_p = parasitic_df["Accuracy (%)"].to_numpy(dtype=float)
_drops_p = parasitic_df["Drop vs Ideal (%)"].to_numpy(dtype=float)
_vdrop_pct = parasitic_df["Estimated_Vdrop_%Vread"].to_numpy(dtype=float)
_baseline_p = float(baseline_acc)
_rp_pos = _rp[_rp > 0]
_accs_pos = _accs_p[_rp > 0]
_vdrop_pos = _vdrop_pct[_rp > 0]
_PARASITIC_BAR_COLORS = ["#718096", "#90CDF4", "#4299E1", "#2C5282", "#1A365D"]
_y_lo = min(_accs_p.min(), _baseline_p) - 0.6
_y_hi = max(_accs_p.max(), _baseline_p) + 0.5

plt.rcParams.update(_PARASITIC_RC)

# Plot 1 — semilog Rp (0.01–10 Ω) + Rp=0 reference
fig1, ax1 = plt.subplots(figsize=(8.5, 5.2))
ax1.semilogx(_rp_pos, _accs_pos, "o-", color="#C05621", linewidth=2.2, markersize=8,
             markeredgecolor="white", markeredgewidth=0.8, label=r"$R_p > 0$", zorder=3)
ax1.axhline(_baseline_p, color="#2C5282", linestyle="--", linewidth=1.4,
            label=rf"$R_p = 0$: {_baseline_p:.2f}%")
ax1.set_xlim(_rp_pos.min() * 0.6, _rp_pos.max() * 1.8)
ax1.set_ylim(_y_lo, _y_hi)
ax1.set_xlabel(r"Row/column parasitic resistance $R_p$ (Ω)")
ax1.set_ylabel("MNIST test accuracy (%)")
ax1.set_title("TNN accuracy vs $R_p$ (log scale, $R_p \\geq 0.01\\,\\Omega$)")
ax1.legend(loc="lower left", framealpha=0.95)
_style_parasitic_ax(ax1)
fig1.tight_layout()
_save_parasitic_fig(fig1, "fig01_accuracy_vs_rp_semilog")

# Plot 2 — linear 0–10 Ω
fig2, ax2 = plt.subplots(figsize=(8.5, 5.2))
ax2.plot(_rp, _accs_p, "o-", color="#C05621", linewidth=2.2, markersize=8,
         markeredgecolor="white", markeredgewidth=0.8)
ax2.set_xlim(-0.3, 10.8)
ax2.set_xticks(_rp)
ax2.set_xticklabels([f"{r:g}" for r in _rp])
ax2.set_ylim(_y_lo, _y_hi)
ax2.set_xlabel(r"Row/column parasitic resistance $R_p$ (Ω)")
ax2.set_ylabel("MNIST test accuracy (%)")
ax2.set_title("TNN accuracy vs $R_p$ (linear scale, 0–10 Ω)")
_style_parasitic_ax(ax2)
fig2.tight_layout()
_save_parasitic_fig(fig2, "fig02_accuracy_vs_rp_linear")

# Plot 3 — accuracy drop bars
fig3, ax3 = plt.subplots(figsize=(8.5, 5.2))
bars = ax3.bar(np.arange(len(_rp)), _drops_p, color=_PARASITIC_BAR_COLORS,
               width=0.62, edgecolor="white", linewidth=0.9)
ax3.set_xticks(np.arange(len(_rp)))
ax3.set_xticklabels([f"{r:g}" for r in _rp])
ax3.set_xlabel(r"$R_p$ (Ω)")
ax3.set_ylabel("Accuracy drop vs ideal (pp)")
ax3.set_title("Accuracy loss due to parasitic interconnect")
drop_hi = max(_drops_p.max() * 1.3, 0.2)
ax3.set_ylim(0, drop_hi)
for bar, drop in zip(bars, _drops_p):
    if drop > 0.01:
        ax3.text(bar.get_x() + bar.get_width() / 2, drop + drop_hi * 0.04,
                 f"{drop:.2f}", ha="center", va="bottom", fontsize=9.5)
_style_parasitic_ax(ax3)
fig3.tight_layout()
_save_parasitic_fig(fig3, "fig03_accuracy_drop_bars")

# Plot 4 — Vdrop log–log
fig4, ax4 = plt.subplots(figsize=(8.5, 5.2))
ax4.loglog(_rp_pos, _vdrop_pos, "s-", color="#2C5282", linewidth=2.2, markersize=8,
           markerfacecolor="#3182CE", markeredgecolor="white", markeredgewidth=0.8)
ax4.set_xlim(_rp_pos.min() * 0.5, _rp_pos.max() * 2.0)
ax4.set_xlabel(r"Row/column parasitic resistance $R_p$ (Ω)")
ax4.set_ylabel(r"Estimated IR drop (% of $V_\mathrm{read}$)")
ax4.set_title(r"Worst-case $V_\mathrm{drop} \propto R_p$ (log–log)")
_style_parasitic_ax(ax4)
fig4.tight_layout()
_save_parasitic_fig(fig4, "fig04_vdrop_vs_rp_loglog")








################################################################################
################################################################################
################################################################################
################################################################################
'''                            Paraisitics Errors                '''
################################################################################
################################################################################
################################################################################
################################################################################



FINAL_RP_OHM = 0.01
FINAL_DRIFT_TIME_SEC = 0.0


print("\n" + "=" * 72)
print("FINAL REAL-CASE TNN SCENARIO")
print("=" * 72)
print(
    f"Peripherals : Quantizer ({TNN_PERIPH_DAC_BITS}b DAC / {TNN_PERIPH_ADC_BITS}b ADC)\n"
    f"Device model: {TNN_APM_MODEL}\n"
    f"Programming : ON   Drift: ON   t={FINAL_DRIFT_TIME_SEC} s\n"
    f"Parasitics  : Rp_row = Rp_col = {FINAL_RP_OHM} Ω"
)

'''
========================================================================
FINAL REAL-CASE TNN SCENARIO
========================================================================
Peripherals : Quantizer (6b DAC / 4b ADC)
Device model: APM_SINW_SBFET
Programming : ON   Drift: ON   t=0.0 s
Parasitics  : Rp_row = Rp_col = 0.01 Ω

'''


final_real_params = build_tnn_apm_params(
    programming_enable=True,
    drift_enable=True,
    time=FINAL_DRIFT_TIME_SEC,
    peripheral_mode="quantizer",
)

# Worst-case parasitic from sweep (Rp = 0.01 Ω)
final_real_params.xbar.array.parasitics.enable = True
final_real_params.xbar.array.parasitics.Rp_row = FINAL_RP_OHM
final_real_params.xbar.array.parasitics.Rp_col = FINAL_RP_OHM
final_real_params.xbar.array.parasitics.current_from_input = True
final_real_params.xbar.array.parasitics.selected_rows = "top"


torch.manual_seed(SEED)
np.random.seed(SEED)

final_real_tnn = from_torch(net_tnn, final_real_params)
final_real_tnn.eval()

y_true_final, pred_final_real, logits_final_real = get_predictions(
    final_real_tnn, test_loader
)
final_real_acc = 100.0 * np.mean(pred_final_real == y_true_final)
final_drop_vs_digital = digital_tnn_acc - final_real_acc
final_mismatch_vs_digital = 100.0 * np.mean(pred_final_real != pred_digital)

print(f"\nDigital TNN accuracy              : {digital_tnn_acc:.2f}%")
print(f"Final real-case accuracy          : {final_real_acc:.2f}%")
print(f"Drop vs digital                   : {final_drop_vs_digital:+.2f} pp")
print(f"Prediction mismatch vs digital    : {final_mismatch_vs_digital:.2f}%")

'''
Digital TNN accuracy              : 93.78%
Final real-case accuracy          : 81.14%
Drop vs digital                   : +12.64 pp
Prediction mismatch vs digital    : 19.61%

'''





# --- Ideal analog reference (same mapping, no prog/drift/quantizer/parasitics) ---
ideal_ref_params = build_tnn_apm_params(
    programming_enable=False,
    drift_enable=False,
    time=0.0,
    peripheral_mode="ideal",
)
ideal_ref_tnn = from_torch(net_tnn, ideal_ref_params)
ideal_ref_tnn.eval()
ideal_ref_acc = evaluate(ideal_ref_tnn, test_loader, device="cpu")

final_real_summary = pd.DataFrame([
    {
        "Case": "Digital TNN",
        "Accuracy (%)": digital_tnn_acc,
        "Drop vs Digital (pp)": 0.0,
        "Mismatch vs Digital (%)": 0.0,
    },
    {
        "Case": "Ideal analog (no non-idealities)",
        "Accuracy (%)": ideal_ref_acc,
        "Drop vs Digital (pp)": digital_tnn_acc - ideal_ref_acc,
        "Mismatch vs Digital (%)": 100.0 * np.mean(
            get_predictions(ideal_ref_tnn, test_loader)[1] != pred_digital
        ),
    },
    {
        "Case": (
            f"Final real case\n"
            f"Quantizer + {TNN_APM_MODEL}\n"
            f"prog/drift + Rp={FINAL_RP_OHM:g}Ω"
        ),
        "Accuracy (%)": final_real_acc,
        "Drop vs Digital (pp)": final_drop_vs_digital,
        "Mismatch vs Digital (%)": final_mismatch_vs_digital,
    },
])

print("\n================ FINAL REAL-CASE SUMMARY ================")
print(final_real_summary.to_string(index=False))

'''
================ FINAL REAL-CASE SUMMARY ================
                                                              Case  Accuracy (%)  Drop vs Digital (pp)  Mismatch vs Digital (%)
                                                       Digital TNN         93.78                  0.00                     0.00
                                  Ideal analog (no non-idealities)         93.78                  0.00                     0.00
Final real case\nQuantizer + APM_SINW_SBFET\nprog/drift + Rp=0.01Ω         81.14                 12.64                    19.61
'''







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


_LAYER_NAMES = ("fc1", "fc2", "fc3")


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


# --- Weight / matrix analysis ---
final_analog_layers = collect_analog_layers(final_real_tnn)
ideal_ref_layers = collect_analog_layers(ideal_ref_tnn)
final_digital_layers = collect_digital_layers(net_tnn)
final_balanced_cores = _collect_tnn_balanced_cores(final_real_tnn)
ideal_balanced_cores = _collect_tnn_balanced_cores(ideal_ref_tnn)

final_weight_vs_digital = _layer_prog_stats(final_analog_layers, final_digital_layers)
final_weight_vs_ideal = _layer_prog_stats(final_analog_layers, ideal_ref_layers)

final_matrix_rows = []
for i, (bc, bc_ref) in enumerate(zip(final_balanced_cores, ideal_balanced_cores)):
    layer_name = _LAYER_NAMES[i] if i < len(_LAYER_NAMES) else f"layer{i + 1}"
    for key, label in (("g_pos", "G_pos"), ("g_neg", "G_neg"), ("g_eff", "G_eff")):
        g_real, g_ref = _align_flat(bc[key], bc_ref[key])
        delta = g_real - g_ref
        final_matrix_rows.append({
            "Layer": layer_name,
            "Matrix": label,
            "MAE vs ideal": float(np.mean(np.abs(delta))),
            "RMSE vs ideal": float(np.sqrt(np.mean(delta ** 2))),
            "r vs ideal": float(np.corrcoef(g_real, g_ref)[0, 1]),
        })

final_matrix_df = pd.DataFrame(final_matrix_rows)

print("\n--- Effective weight statistics (programmed analog vs digital ternary) ---")
for st in final_weight_vs_digital:
    print(
        f"  {st['layer_name']:4s}  MAE={st['mae']:.4e}  "
        f"RMSE={st['rmse']:.4e}  r(digital)={st['corr']:.4f}"
    )

print("\n--- Effective weight statistics (programmed analog vs ideal analog mapping) ---")
for st in final_weight_vs_ideal:
    print(
        f"  {st['layer_name']:4s}  MAE={st['mae']:.4e}  "
        f"RMSE={st['rmse']:.4e}  r(ideal)={st['corr']:.4f}"
    )

print("\n--- Balanced conductance matrices (G_pos / G_neg / G_eff vs ideal) ---")
print(final_matrix_df.to_string(index=False))


'''
--- Effective weight statistics (programmed analog vs digital ternary) ---
  fc1   MAE=3.0888e-02  RMSE=3.8551e-02  r(digital)=0.9959
  fc2   MAE=3.2878e-02  RMSE=4.0907e-02  r(digital)=0.9975
  fc3   MAE=4.0606e-02  RMSE=4.9058e-02  r(digital)=0.9987

--- Effective weight statistics (programmed analog vs ideal analog mapping) ---
  fc1   MAE=3.0888e-02  RMSE=3.8551e-02  r(ideal)=0.9959
  fc2   MAE=3.2878e-02  RMSE=4.0907e-02  r(ideal)=0.9975
  fc3   MAE=4.0606e-02  RMSE=4.9058e-02  r(ideal)=0.9987

--- Balanced conductance matrices (G_pos / G_neg / G_eff vs ideal) ---
Layer Matrix  MAE vs ideal  RMSE vs ideal  r vs ideal
  fc1  G_pos  2.702960e-11   3.006835e-11    0.996276
  fc1  G_neg  2.755623e-11   3.035677e-11    0.994444
  fc1  G_eff  1.492705e-11   1.863028e-11    0.995854
  fc2  G_pos  2.719698e-11   3.016431e-11    0.995740
  fc2  G_neg  2.537367e-11   2.904700e-11    0.997941
  fc2  G_eff  1.588876e-11   1.976889e-11    0.997523
  fc3  G_pos  2.728425e-11   3.028773e-11    0.996251
  fc3  G_neg  1.769202e-11   2.361859e-11    0.998987
  fc3  G_eff  1.962371e-11   2.370831e-11    0.998665
  
  '''






# --- Academic figures ---
try:
    _FINAL_RC = _PERIPH_RC
except NameError:
    _FINAL_RC = {"font.family": "serif", "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13}


def _save_final_fig(fig, stem: str) -> None:
    plt.show()


# Figure 1 — accuracy bar chart (digital vs ideal vs final real case)
_final_cases = ["Digital\nTNN", "Ideal\nanalog", "Final\nreal case"]
_final_accs = [
    float(digital_tnn_acc),
    float(ideal_ref_acc),
    float(final_real_acc),
]
_final_colors = ["#2C5282", "#718096", "#C05621"]

with plt.rc_context(_FINAL_RC):
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    bars = ax.bar(
        np.arange(3), _final_accs, color=_final_colors,
        width=0.58, edgecolor="white", linewidth=0.9, zorder=3,
    )
    ax.axhline(
        digital_tnn_acc, color="#2C5282", linestyle="--", linewidth=1.2,
        alpha=0.65, label=f"Digital baseline ({digital_tnn_acc:.2f}%)", zorder=1,
    )
    y_lo = min(_final_accs) - 1.0
    y_hi = max(_final_accs) + 1.2
    ax.set_ylim(0, y_hi)
    ax.set_xticks(np.arange(3), _final_cases)
    ax.set_ylabel("MNIST test accuracy (%)")
    ax.set_title(
    "Final real-case TNN accuracy\n"
    f"Quantizer {TNN_PERIPH_DAC_BITS}b/{TNN_PERIPH_ADC_BITS}b, "
    f"{TNN_APM_MODEL}, prog+drift, $R_p={FINAL_RP_OHM:g}\\,\\Omega$",
    pad=20,   # increase this value
    )
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.45, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, acc in zip(bars, _final_accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, acc + 0.08,
            f"{acc:.2f}%", ha="center", va="bottom", fontsize=9.5,
        )
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    _save_final_fig(fig, "fig01_final_accuracy_bar")
    
    
    
    
    
    
    

# Figure 2 — per-layer weight MAE (digital vs ideal reference)
_layer_labels = [st["layer_name"] for st in final_weight_vs_digital]
_mae_digital = [st["mae"] for st in final_weight_vs_digital]
_mae_ideal = [st["mae"] for st in final_weight_vs_ideal]
_x_layers = np.arange(len(_layer_labels))
_w = 0.36

with plt.rc_context(_FINAL_RC):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.bar(
        _x_layers - _w / 2, _mae_digital, _w,
        label="vs digital ternary", color="#3182CE", edgecolor="white", linewidth=0.8,
    )
    ax.bar(
        _x_layers + _w / 2, _mae_ideal, _w,
        label="vs ideal analog mapping", color="#C05621", edgecolor="white", linewidth=0.8,
    )
    ax.set_xticks(_x_layers, _layer_labels)
    ax.set_ylabel("Mean absolute error (effective weight)")
    ax.set_title("Final real-case programmed weights — per-layer MAE")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.45, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    _save_final_fig(fig, "fig02_final_weight_mae_by_layer")

# Figure 3 — confusion matrix
cm_final_real = confusion_matrix_np(y_true_final, pred_final_real)
plot_confusion_matrix_academic(
    cm_final_real,
    (
        f"Final real case — confusion matrix\n"
        f"Quantizer, prog+drift, $R_p={FINAL_RP_OHM:g}\\,\\Omega$"
    ),
    "fig03_cm_final_real_case",
)









#----------FINAL DETERMINISTIC--------
_final_cases = ["Digital TNN", "Ideal Analog", "With Programming\nError" , "With Quantizer\n ADC|DAC Error", "Final\nReal case"]
_final_accs = [
    float(digital_tnn_acc),
    float(ideal_ref_acc),
    92.56,
    81.14,
    float(final_real_acc),
]

_final_colors = [
    "#2C5282",  # dark academic blue
    "#718096",  # blue-gray
    "#C05621",  # burnt orange
    "#2B6CB0",  # vivid medium blue
    "#2F855A",  # academic green
]


with plt.rc_context(_FINAL_RC):
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    bars = ax.bar(
        np.arange(5), _final_accs, color=_final_colors,
        width=0.58, edgecolor="white", linewidth=0.9, zorder=3,
    )
    ax.axhline(
        digital_tnn_acc, color="#2C5282", linestyle="--", linewidth=1.2,
        alpha=0.65, label=f"Digital baseline ({digital_tnn_acc:.2f}%)", zorder=1,
    )
    y_lo = min(_final_accs) - 1.0
    y_hi = max(_final_accs) + 1.2
    ax.set_ylim(0, y_hi)
    ax.set_xticks(np.arange(5), _final_cases)
    ax.set_ylabel("MNIST test accuracy (%)")
    ax.set_title(
    "Final real-case TNN accuracy\n"
    f"Quantizer {TNN_PERIPH_DAC_BITS}b/{TNN_PERIPH_ADC_BITS}b, "
    f"{TNN_APM_MODEL}, programming error, $R_p={FINAL_RP_OHM:g}\\,\\Omega$",
    pad=20,   # increase this value
    )
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.45, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, acc in zip(bars, _final_accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, acc + 0.08,
            f"{acc:.2f}%", ha="center", va="bottom", fontsize=9.5,
        )
    fig.tight_layout()
    _save_final_fig(fig, "fig01_final_accuracy_bar")
    








