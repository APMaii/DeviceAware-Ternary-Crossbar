from __future__ import annotations

'''
In The Name of God

Ali Pilehvar Meibody

Last Update : 25 May 2026



10_Device_Aware_Training.py


In previous file we consider more real senario with considering ADC/DAC and then
resistance (paraisicic errors) and finally we reach our best final which is programmed (apm)
drift errors and also Quantizer 4b/6b and paraisitcs 

and here we want to recovery that loss that we have

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

from paths import DATA_DIR, PTH_DIR, PROJECT_DIR, choose_checkpoint, ensure_cross_sim_on_path
ensure_cross_sim_on_path()

from simulator import CrossSimParameters
from simulator.parameters.xbar_parameters import ADCRangeLimits
from simulator.algorithms.dnn.torch.convert import from_torch
from simulator.algorithms.dnn.torch.convert import convertible_modules
import pandas as pd
import copy
from pathlib import Path
from tqdm import tqdm
from simulator.algorithms.dnn.torch.convert import from_torch, synchronize
import os
import random
from datetime import date
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms








# ============================================
'''                Variables              '''
# ============================================
SEED = 42  # change this to try other runs; keep fixed for identical results

batch_size = 64
BATCH_SIZE = batch_size

#learning_rate = 0.0001 #for ANN
learning_rate=1e-4 #for TNN
LEARNING_RATE = learning_rate
DAT_LR=learning_rate


num_epochs = 26
NUM_EPOCHS = num_epochs
DAT_EPOCHS = num_epochs


SAVE_FIGS = False
SAVE_CHECKPOINTS = False



DAT_DIR = PTH_DIR / "Device_aware"
DAT_DIR.mkdir(parents=True, exist_ok=True)


FIGURE_DPI = 150
THRESHOLD = 0.05
SMOOTH_TW_WIDTH = 0.7
DEVICE = torch.device("cpu")




#These are our BASE MODELS
# Base models are chosen from Pth_Models with choose_checkpoint("ann") / ("tnn").




R_MIN=1.410360e+09  
R_MAX=4.429246e+09 

TNN_APM_MODEL = "APM_SINW_SBFET"


#comes from 09 
DAC_BITS =6
ADC_BITS=4
INPUT_RANGE=32
ADC_RANGE=1



date='07sep'



#=========================================================
'''                  Load Data               '''
#=========================================================

def set_seed(seed: int = SEED) -> None:
    """Set seeds so weight init, shuffling, and training are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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



def evaluate_acc(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            out = model(x)
            pred = out.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    return 100.0 * correct / total






################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
'''                             Device-Aware Training                 '''
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################


# TNN — smooth threshold-window surrogate (w=0.7)
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







FINAL_RP_OHM = 0.01
FINAL_DRIFT_TIME_SEC = 0.0



def train_one_epoch_digital(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0.0

    for x, y in train_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)



train_loader, test_loader = make_loaders()





# ============================================================
# 1. Digital TNN training
# ============================================================

'''
Here we have TNN which is ternary neural network with TernaryMNIST() class
and we used to train that with 20 epochs (not 100 epochs)

Here , this is all Digital and on pytorch
no non ideality , no cross sim , no other things, jusst for our case



'''
set_seed(SEED)




digital_tnn = TernaryMNIST().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer_digital = torch.optim.Adam(digital_tnn.parameters(), lr=DAT_LR)

digital_loss_history = []
digital_acc_history = []

print("\n" + "=" * 72)
print("TRAINING DIGITAL TNN")
print("=" * 72)


for epoch in tqdm(range(DAT_EPOCHS), desc="Digital TNN"):
    loss = train_one_epoch_digital(
        digital_tnn,
        train_loader,
        criterion,
        optimizer_digital
    )

    acc = evaluate_acc(digital_tnn, test_loader)

    digital_loss_history.append(loss)
    digital_acc_history.append(acc)

    print(
        f"Epoch {epoch+1:03d}/{DAT_EPOCHS} | "
        f"loss = {loss:.5f} | "
        f"test acc = {acc:.2f}%"
    )


digital_final_acc = evaluate_acc(digital_tnn, test_loader)






np.save(DAT_DIR / f"{date}_digital_loss_history.npy", np.array(digital_loss_history))
np.save(DAT_DIR / f"{date}_digital_acc_history.npy", np.array(digital_acc_history))

torch.save(
    {
        "state_dict": digital_tnn.state_dict(),
        "final_accuracy": digital_final_acc,
        "loss_history": digital_loss_history,
        "acc_history": digital_acc_history,
    },
    DAT_DIR / f"{date}_digital_tnn.pt",
)







'''
========================================================================
TRAINING DIGITAL TNN
========================================================================
Digital TNN:   4%|▍         | 1/26 [00:05<02:27,  5.92s/it]Epoch 001/26 | loss = 2.40890 | test acc = 73.79%
Digital TNN:   8%|▊         | 2/26 [00:11<02:20,  5.86s/it]Epoch 002/26 | loss = 0.96564 | test acc = 79.86%
Digital TNN:  12%|█▏        | 3/26 [00:17<02:15,  5.89s/it]Epoch 003/26 | loss = 0.78713 | test acc = 83.35%
Digital TNN:  15%|█▌        | 4/26 [00:23<02:11,  5.97s/it]Epoch 004/26 | loss = 0.74182 | test acc = 79.67%
Digital TNN:  19%|█▉        | 5/26 [00:29<02:04,  5.92s/it]Epoch 005/26 | loss = 0.69051 | test acc = 86.04%
Digital TNN:  23%|██▎       | 6/26 [00:35<01:58,  5.93s/it]Epoch 006/26 | loss = 0.63772 | test acc = 88.19%
Digital TNN:  27%|██▋       | 7/26 [00:41<01:52,  5.93s/it]Epoch 007/26 | loss = 0.58349 | test acc = 86.98%
Digital TNN:  31%|███       | 8/26 [00:47<01:47,  5.96s/it]Epoch 008/26 | loss = 0.54627 | test acc = 87.27%
Digital TNN:  35%|███▍      | 9/26 [00:53<01:41,  5.97s/it]Epoch 009/26 | loss = 0.52612 | test acc = 87.93%
Digital TNN:  38%|███▊      | 10/26 [00:59<01:35,  5.99s/it]Epoch 010/26 | loss = 0.51860 | test acc = 90.06%
Digital TNN:  42%|████▏     | 11/26 [01:05<01:29,  5.99s/it]Epoch 011/26 | loss = 0.51876 | test acc = 88.88%
Digital TNN:  46%|████▌     | 12/26 [01:11<01:24,  6.00s/it]Epoch 012/26 | loss = 0.50005 | test acc = 86.87%
Digital TNN:  50%|█████     | 13/26 [01:17<01:18,  6.00s/it]Epoch 013/26 | loss = 0.44383 | test acc = 89.75%
Digital TNN:  54%|█████▍    | 14/26 [01:23<01:11,  5.99s/it]Epoch 014/26 | loss = 0.44475 | test acc = 87.88%
Digital TNN:  58%|█████▊    | 15/26 [01:29<01:06,  6.01s/it]Epoch 015/26 | loss = 0.42334 | test acc = 91.59%
Digital TNN:  62%|██████▏   | 16/26 [01:35<00:59,  6.00s/it]Epoch 016/26 | loss = 0.41854 | test acc = 91.82%
Digital TNN:  65%|██████▌   | 17/26 [01:41<00:54,  6.02s/it]Epoch 017/26 | loss = 0.41770 | test acc = 90.56%
Digital TNN:  69%|██████▉   | 18/26 [01:47<00:48,  6.00s/it]Epoch 018/26 | loss = 0.38763 | test acc = 92.38%
Digital TNN:  73%|███████▎  | 19/26 [01:53<00:42,  6.01s/it]Epoch 019/26 | loss = 0.39942 | test acc = 90.79%
Digital TNN:  77%|███████▋  | 20/26 [01:59<00:36,  6.01s/it]Epoch 020/26 | loss = 0.37337 | test acc = 92.58%
Digital TNN:  81%|████████  | 21/26 [02:05<00:30,  6.01s/it]Epoch 021/26 | loss = 0.38024 | test acc = 92.10%
Digital TNN:  85%|████████▍ | 22/26 [02:11<00:24,  6.03s/it]Epoch 022/26 | loss = 0.34792 | test acc = 90.95%
Digital TNN:  88%|████████▊ | 23/26 [02:17<00:18,  6.01s/it]Epoch 023/26 | loss = 0.36255 | test acc = 91.29%
Digital TNN:  92%|█████████▏| 24/26 [02:23<00:12,  6.05s/it]Epoch 024/26 | loss = 0.35899 | test acc = 89.95%
Digital TNN:  96%|█████████▌| 25/26 [02:29<00:06,  6.05s/it]Epoch 025/26 | loss = 0.33670 | test acc = 90.54%
Digital TNN: 100%|██████████| 26/26 [02:35<00:00,  6.00s/it]Epoch 026/26 | loss = 0.32619 | test acc = 91.53%


'''



################################################################
################################################################
# ============================================================
# 2. : Device-aware TNN training
# ===========================================================

'''
Here we want to consider the ternary layer inside Cross sim



in torch/linear.py we add AnalogTernaryLinear and this is something

import torch
class AnalogTernaryLinear(AnalogLinear):
    def __init__(
        self,
        params,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        bias_rows=0,
        threshold=0.05,
        width=0.7,
    ):
        
    ......
    ......
    def ternary_weight_cpu(self):
    def form_matrix(self):
    @classmethod
    def from_torch(cls, layer, params, bias_rows=0):
    
        
    
Our existing AnalogLinear uses CrossSim forward and ideal backward through 
AnalogLinearGrad; this new version changes the forward matrix to ternary and 
changes backward to surrogate-gradient behavior.



Below this we add AnalogTernaryLinearGrad 

class AnalogTernaryLinearGrad(Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        core,
        bias_rows,
        training,
        threshold,
        width,
    ):
    @staticmethod
    def backward(ctx, grad_output):
        
    
Now we have 
Forward: CrossSim analog MVM with ternary weights
Backward: surrogate gradient through full-precision shadow weights





NO w in convert.py

from simulator.algorithms.dnn.torch.linear import AnalogLinear, AnalogTernaryLinear

_conversion_map = {
    Linear: AnalogLinear,
    TernaryLinear: AnalogTernaryLinear,
    Conv1d: AnalogConv1d,
    Conv2d: AnalogConv2d,
    Conv3d: AnalogConv3d,
}



also we build TernaryLinear in tnn_layer.py 






What happening inside CrossSim? ------------------------------------
In each loop , after from_torch. TernayrLinear --> AnalogTernaryLinear
from_matrix() ternarizes shadow weights to {-1,0,+1} before programming
set_matrix() applies programming(write) error via device.apply_write_error()
the core stores teh noisy programmed conductances


Forward (model.train() --> training=True)
AnalogTernaryLinearGrad.forward calls core.,aapply(x.detach(()))
which goes through the full analog MVM path.
Effect	In forward?
Ternary programmed weights
Yes (from last synchronize / set_matrix)
Programming error
Yes (baked in at program time; re-sampled on each synchronize)
DAC (input quantization)
Yes, if enabled in final_real_params.xbar.dac
ADC (output quantization)
Yes, if enabled in final_real_params.xbar.adc
Read noise
Yes, new noise each forward (if enabled)
Parasitics
Yes, if enabled

So forward is not ideal digital F.linear — it is the CrossSim analog path with ternary values programmed into the array



Backward (loss.backward())
Effect	In backward?
STE / smooth surrogate on shadow weight
Yes (grad_surrogate in AnalogTernaryLinearGrad)
grad_x
Ideal grad_output @ W_core
W_core for grad_x
From get_core_weights() → programmed matrix with write error, without per-forward read noise
ADC/DAC gradients
No (CrossSim torch backward is ideal, per package docs)
Read noise in backward
No


So backprop is not a full differentiable model of ADC/DAC/read noise. It is:

Forward: full non-ideal analog simulation
Backward: ideal linear grads + STE through ternary thresholds
That is the intended CrossSim-in-the-loop tradeoff, with your extra ternary STE on top.

4. After optimizer.step() + synchronize(model)
Optimizer updates continuous shadow weight in place.
synchronize() → form_matrix() re-ternarizes → set_matrix() reprograms.
Programming error is applied again on the new ternary pattern.
Without synchronize, shadow weights move but the array still holds the old ternary program — training would be wrong.





'''

from simulator import CrossSimParameters
from simulator.algorithms.dnn.torch.convert import from_torch
from simulator.algorithms.dnn.torch.convert import convertible_modules


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








#**********
def train_one_epoch_device_aware_tnn(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0.0

    for x, y in train_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        # Forward uses CrossSim analog core with ternary programmed weights
        out = model(x)

        loss = criterion(out, y)

        # Backward uses your surrogate/STE gradient inside AnalogTernaryLinearGrad
        loss.backward()

        optimizer.step()

        # Required after optimizer.step() because optimizer updates weights in-place
        synchronize(model)

        total_loss += loss.item()
        
    #inja mitonim bzarim synchrnoize ro 
    #synchronize(model)

    return total_loss / len(train_loader)





# ============================================================
# Build TernaryMNIST and convert directly to CrossSim
# ============================================================



'''
Or we can get the weights from Base TNN from 100 Epochs
and then


from pathlib import Path
import sys

#from 04_Load_ANN_TNN.py import load_ann , load_tnn_ternary , print_model_summary
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
import importlib
_load = importlib.import_module(f"04_Load_ANN_TNN")
load_ann = _load.load_ann
load_tnn_ternary = _load.load_tnn_ternary
print_model_summary = _load.print_model_summary



ANN_CHECKPOINT = choose_checkpoint("ann")
TNN_CHECKPOINT = choose_checkpoint("tnn")


net_ann, ann_path, ann_meta = load_ann(ANN_CHECKPOINT)
print_model_summary(net_ann, "ANN (full-precision)", ann_path, ann_meta)


net_tnn, tnn_path, tnn_meta = load_tnn_ternary(TNN_CHECKPOINT)
print_model_summary(net_tnn, "TNN (ternary -1/0/+1)", tnn_path, tnn_meta)



then we can use net_tnn weights and see in 4,5 epochs what happenings
 





consider that for first we have 84 , which is ok and is best , but we want
to do something like fine-tuning or anythign to  go over 84 % and something


'''


train_loader_dat, test_loader = make_loaders()

set_seed(SEED)



from simulator.algorithms.dnn.torch.tnn_layer import TernaryLinear



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

base_tnn_for_dat = TernaryMNIST().to(DEVICE)


print("\nConvertible modules before CrossSim conversion:")
print(convertible_modules(base_tnn_for_dat))


#we call the function from 09 with all True and also quantizer
final_real_params = build_tnn_apm_params(
    programming_enable=True,
    drift_enable=True,
    time=FINAL_DRIFT_TIME_SEC,
    peripheral_mode="quantizer",
)


device_aware_tnn = from_torch(base_tnn_for_dat, final_real_params)
device_aware_tnn.to(DEVICE)
device_aware_tnn.train()

print("\nAnalog modules after conversion:")
print(device_aware_tnn)






# ============================================================
# Train
# ============================================================
criterion = nn.CrossEntropyLoss()
optimizer_dat = torch.optim.Adam(device_aware_tnn.parameters(), lr=DAT_LR)

dat_loss_history = []
dat_acc_history = []

print("\n" + "=" * 72)
print("TRAINING DEVICE-AWARE TERNARY MNIST")
print("=" * 72)

initial_dat_acc = evaluate_acc(device_aware_tnn, test_loader)
print(f"Initial device-aware ternary analog accuracy: {initial_dat_acc:.2f}%")



for epoch in tqdm(range(DAT_EPOCHS), desc="Device-aware TNN"):
    loss = train_one_epoch_device_aware_tnn(
        model=device_aware_tnn,
        train_loader=train_loader_dat,
        criterion=criterion,
        optimizer=optimizer_dat,
    )

    acc = evaluate_acc(device_aware_tnn, test_loader)

    dat_loss_history.append(loss)
    dat_acc_history.append(acc)

    print(
        f"Epoch {epoch + 1:03d}/{DAT_EPOCHS} | "
        f"loss = {loss:.5f} | "
        f"test acc = {acc:.2f}%"
    )

dat_final_acc = evaluate_acc(device_aware_tnn, test_loader)

print("\n" + "=" * 72)
print("DEVICE-AWARE TERNARY TRAINING SUMMARY")
print("=" * 72)
print(f"Final device-aware ternary accuracy: {dat_final_acc:.2f}%")



'''


========================================================================
TRAINING DEVICE-AWARE TERNARY MNIST
========================================================================
Initial device-aware ternary analog accuracy: 9.58%
Device-aware TNN:   4%|▍         | 1/26 [00:36<15:18, 36.75s/it]Epoch 001/26 | loss = 11.20437 | test acc = 15.46%
Device-aware TNN:   8%|▊         | 2/26 [01:13<14:43, 36.80s/it]Epoch 002/26 | loss = 6.36785 | test acc = 29.01%
Device-aware TNN:  12%|█▏        | 3/26 [01:48<13:42, 35.75s/it]Epoch 003/26 | loss = 3.85591 | test acc = 46.81%
Device-aware TNN:  15%|█▌        | 4/26 [02:23<13:02, 35.56s/it]Epoch 004/26 | loss = 2.85285 | test acc = 49.03%
Device-aware TNN:  19%|█▉        | 5/26 [02:58<12:21, 35.32s/it]Epoch 005/26 | loss = 2.84958 | test acc = 50.16%
Device-aware TNN:  23%|██▎       | 6/26 [03:32<11:41, 35.06s/it]Epoch 006/26 | loss = 2.11708 | test acc = 63.58%
Device-aware TNN:  27%|██▋       | 7/26 [04:07<11:03, 34.92s/it]Epoch 007/26 | loss = 1.51818 | test acc = 78.95%
Device-aware TNN:  31%|███       | 8/26 [04:41<10:24, 34.67s/it]Epoch 008/26 | loss = 1.48753 | test acc = 84.98%
Device-aware TNN:  35%|███▍      | 9/26 [05:16<09:50, 34.74s/it]Epoch 009/26 | loss = 1.25458 | test acc = 84.63%
Device-aware TNN:  38%|███▊      | 10/26 [05:51<09:16, 34.80s/it]Epoch 010/26 | loss = 1.03698 | test acc = 85.54%
Device-aware TNN:  42%|████▏     | 11/26 [06:26<08:42, 34.83s/it]Epoch 011/26 | loss = 0.94013 | test acc = 88.56%
Device-aware TNN:  46%|████▌     | 12/26 [07:01<08:10, 35.04s/it]Epoch 012/26 | loss = 1.10242 | test acc = 81.16%
Device-aware TNN:  50%|█████     | 13/26 [07:37<07:36, 35.11s/it]Epoch 013/26 | loss = 0.87509 | test acc = 79.91%
Device-aware TNN:  54%|█████▍    | 14/26 [08:12<07:01, 35.12s/it]
Device-aware TNN:  58%|█████▊    | 15/26 [08:47<06:27, 35.23s/it]Epoch 015/26 | loss = 1.03957 | test acc = 83.10%
Device-aware TNN:  62%|██████▏   | 16/26 [09:22<05:51, 35.11s/it]Epoch 016/26 | loss = 0.88803 | test acc = 80.47%
Device-aware TNN:  65%|██████▌   | 17/26 [09:57<05:16, 35.16s/it]Epoch 017/26 | loss = 0.73583 | test acc = 91.08%
Device-aware TNN:  69%|██████▉   | 18/26 [10:32<04:40, 35.07s/it]Epoch 018/26 | loss = 0.64212 | test acc = 90.40%
Device-aware TNN:  73%|███████▎  | 19/26 [11:07<04:05, 35.01s/it]Epoch 019/26 | loss = 0.60039 | test acc = 89.37%
Device-aware TNN:  77%|███████▋  | 20/26 [11:42<03:30, 35.12s/it]Epoch 020/26 | loss = 0.57494 | test acc = 91.60%
Device-aware TNN:  81%|████████  | 21/26 [12:18<02:56, 35.28s/it]Epoch 021/26 | loss = 0.56086 | test acc = 90.69%
Device-aware TNN:  85%|████████▍ | 22/26 [12:53<02:21, 35.32s/it]Epoch 022/26 | loss = 0.53661 | test acc = 90.31%
Device-aware TNN:  88%|████████▊ | 23/26 [13:29<01:46, 35.36s/it]Epoch 023/26 | loss = 0.53249 | test acc = 91.88%
Device-aware TNN:  92%|█████████▏| 24/26 [14:04<01:10, 35.19s/it]Epoch 024/26 | loss = 0.51420 | test acc = 91.64%
Device-aware TNN:  96%|█████████▌| 25/26 [14:39<00:35, 35.21s/it]Epoch 025/26 | loss = 0.46408 | test acc = 91.05%
Device-aware TNN: 100%|██████████| 26/26 [15:14<00:00, 35.16s/it]Epoch 026/26 | loss = 0.47253 | test acc = 93.68%


========================================================================
DEVICE-AWARE TERNARY TRAINING SUMMARY
========================================================================
Final device-aware ternary accuracy: 93.68%







'''

# ============================================================
# Save
# CrossSim analog layers should not be directly torch.save'd long-term.
# Save histories and convert to normal torch if needed.
# ============================================================


np.save(DAT_DIR / f"{date}_device_aware_ternary_loss_history.npy", np.array(dat_loss_history))
np.save(DAT_DIR / f"{date}_device_aware_ternary_acc_history.npy", np.array(dat_acc_history))



torch.save(
    {
        "state_dict": device_aware_tnn.state_dict(),
        "final_accuracy": dat_final_acc,
        "loss_history": dat_loss_history,
        "acc_history": dat_acc_history,
    },
    DAT_DIR / f"{date}_device_aware_tnn.pt",
)











# ============================================================
# Visualize ternary shadow weights vs analog core weights
# ============================================================

# ============================================================
# Professional visualization:
# Shadow weights vs Ternary states vs Analog core
# ============================================================



from simulator.algorithms.dnn.torch.convert import analog_modules
import torch
import numpy as np

def inspect_analog_tnn_weights(model, threshold=0.05):
    for i, layer in enumerate(analog_modules(model)):
        print("\n" + "=" * 70)
        print(f"Layer {i}: {layer.__class__.__name__}")
        print("=" * 70)

        # 1. Trainable full-precision shadow weights
        shadow_w = layer.weight.detach().cpu()

        shadow_unique = torch.unique(shadow_w)
        print("Shadow weight shape:", tuple(shadow_w.shape))
        print("Shadow weight min/max:", shadow_w.min().item(), shadow_w.max().item())
        print("Shadow unique count:", shadow_unique.numel())

        # 2. Ideal ternary version made from shadow weights
        ternary_w = torch.where(
            shadow_w > threshold,
            torch.ones_like(shadow_w),
            torch.where(
                shadow_w < -threshold,
                -torch.ones_like(shadow_w),
                torch.zeros_like(shadow_w),
            )
        )

        vals, counts = torch.unique(ternary_w, return_counts=True)
        print("Ideal ternary values/counts:")
        for v, c in zip(vals, counts):
            print(f"  {float(v): .1f}: {int(c)}")

        # 3. CrossSim programmed/core weights
        core_w, core_b = layer.get_core_weights()
        core_w = core_w.detach().cpu()

        print("Core weight shape:", tuple(core_w.shape))
        print("Core weight min/max:", core_w.min().item(), core_w.max().item())

        core_unique_rounded = torch.unique(torch.round(core_w, decimals=4))
        print("Rounded core unique values:", core_unique_rounded[:20])
        print("Rounded core unique count:", core_unique_rounded.numel())

        # 4. Compare ternary ideal vs core weights
        diff = core_w - ternary_w
        print("Mean abs(core - ternary):", diff.abs().mean().item())
        print("Max abs(core - ternary):", diff.abs().max().item())
        
     
inspect_analog_tnn_weights(device_aware_tnn, threshold=THRESHOLD)
        
        
       
'''
======================================================================
Layer 0: AnalogTernaryLinear
======================================================================
Shadow weight shape: (256, 784)
Shadow weight min/max: -0.29612693190574646 0.12136330455541611
Shadow unique count: 200369
Ideal ternary values/counts:
  -1.0: 4113
   0.0: 189134
   1.0: 7457
Core weight shape: (256, 784)
Core weight min/max: -1.1116403341293335 1.1116403341293335
Rounded core unique values: tensor([-1.1116, -1.1112, -1.1111, -1.1109, -1.1106, -1.1104, -1.1100, -1.1099,
        -1.1098, -1.1092, -1.1088, -1.1087, -1.1082, -1.1080, -1.1079, -1.1077,
        -1.1074, -1.1072, -1.1067, -1.1066])
Rounded core unique count: 5113
Mean abs(core - ternary): 0.02955351024866104
Max abs(core - ternary): 0.15903052687644958

======================================================================
Layer 1: AnalogTernaryLinear
======================================================================
Shadow weight shape: (128, 256)
Shadow weight min/max: -0.18044385313987732 0.1030857190489769
Shadow unique count: 32756
Ideal ternary values/counts:
  -1.0: 5187
   0.0: 26699
   1.0: 882
Core weight shape: (128, 256)
Core weight min/max: -1.1116403341293335 1.1116403341293335
Rounded core unique values: tensor([-1.1116, -1.1114, -1.1113, -1.1112, -1.1111, -1.1108, -1.1107, -1.1106,
        -1.1104, -1.1103, -1.1102, -1.1101, -1.1100, -1.1096, -1.1095, -1.1093,
        -1.1091, -1.1089, -1.1085, -1.1083])
Rounded core unique count: 3934
Mean abs(core - ternary): 0.03182905912399292
Max abs(core - ternary): 0.1451931595802307

======================================================================
Layer 2: AnalogTernaryLinear
======================================================================
Shadow weight shape: (10, 128)
Shadow weight min/max: -0.5475693941116333 0.14360670745372772
Shadow unique count: 1280
Ideal ternary values/counts:
  -1.0: 308
   0.0: 773
   1.0: 199
Core weight shape: (10, 128)
Core weight min/max: -1.1116403341293335 1.1116403341293335
Rounded core unique values: tensor([-1.1116, -1.1100, -1.1079, -1.1072, -1.1020, -1.1012, -1.1008, -1.0997,
        -1.0987, -1.0964, -1.0961, -1.0935, -1.0929, -1.0928, -1.0926, -1.0906,
        -1.0900, -1.0899, -1.0894, -1.0890])
Rounded core unique count: 1033
Mean abs(core - ternary): 0.03662445396184921
Max abs(core - ternary): 0.11755058914422989


'''

from simulator.algorithms.dnn.torch.convert import analog_modules
import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_ternary_vs_core_academic(model, threshold=0.05):

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.titlesize": 15,
        "axes.titleweight": "bold",
        "figure.titleweight": "bold",
    })

    layers = analog_modules(model)

    for i, layer in enumerate(layers):

        # ----------------------------------------------------
        # Extract weights
        # ----------------------------------------------------
        shadow_w = layer.weight.detach().cpu().flatten()

        ternary_w = torch.where(
            shadow_w > threshold,
            torch.ones_like(shadow_w),
            torch.where(
                shadow_w < -threshold,
                -torch.ones_like(shadow_w),
                torch.zeros_like(shadow_w),
            )
        )

        core_w, _ = layer.get_core_weights()
        core_w = core_w.detach().cpu().flatten()

        # ----------------------------------------------------
        # Figure
        # ----------------------------------------------------
        fig = plt.figure(figsize=(15, 4.8), dpi=260)

        # ====================================================
        # 1. Shadow weights
        # ====================================================
        ax1 = plt.subplot(1, 3, 1)

        ax1.hist(
            shadow_w.numpy(),
            bins=140,
            density=True,
            alpha=0.92,
            color="#1F3B73",   # deep academic blue
            edgecolor="black",
            linewidth=0.25,
        )

        ax1.axvline(
            threshold,
            linestyle="--",
            linewidth=1.5,
            color="#8B0000",
        )

        ax1.axvline(
            -threshold,
            linestyle="--",
            linewidth=1.5,
            color="#8B0000",
        )

        ax1.set_title("Shadow Weights")
        ax1.set_xlabel("Weight Value")
        ax1.set_ylabel("Density")

        ax1.grid(alpha=0.22)

        # ====================================================
        # 2. Ternary states
        # ====================================================
        ax2 = plt.subplot(1, 3, 2)

        vals, counts = torch.unique(ternary_w, return_counts=True)

        ax2.bar(
            vals.numpy(),
            counts.numpy(),
            width=0.55,
            color="#B08D57",   # muted academic gold
            edgecolor="black",
            linewidth=0.6,
        )

        ax2.set_xticks([-1, 0, 1])

        ax2.set_title("Ideal Ternary States")
        ax2.set_xlabel("Ternary State")
        ax2.set_ylabel("Count")

        ax2.grid(alpha=0.22)

        # ====================================================
        # 3. Analog core weights
        # ====================================================
        ax3 = plt.subplot(1, 3, 3)

        ax3.hist(
            core_w.numpy(),
            bins=160,
            density=True,
            alpha=0.92,
            color="#5A5A5A",   # dark gray
            edgecolor="black",
            linewidth=0.2,
        )

        ax3.axvline(
            1,
            linestyle="--",
            linewidth=1.5,
            color="#B08D57",
        )

        ax3.axvline(
            -1,
            linestyle="--",
            linewidth=1.5,
            color="#B08D57",
        )

        ax3.axvline(
            0,
            linestyle="--",
            linewidth=1.3,
            color="#1F3B73",
        )

        ax3.set_title("Analog Core Weights")
        ax3.set_xlabel("Physical Weight / Conductance")
        ax3.set_ylabel("Density")

        ax3.grid(alpha=0.22)

        # ====================================================
        # Overall title
        # ====================================================
        plt.suptitle(
            f"Device-Aware Ternary Weight Distribution — Layer {i}",
            y=1.03,
        )

        plt.tight_layout()
        plt.show()


# ============================================================
# Run
# ============================================================

plot_ternary_vs_core_academic(
    device_aware_tnn,
    threshold=THRESHOLD,
)
        
        
        
        
        






# ============================================================
# 3. Summary
# ============================================================

print("\n" + "=" * 72)
print("DIGITAL vs DEVICE-AWARE TRAINING SUMMARY")
print("=" * 72)
print(f"Digital TNN final accuracy       : {digital_final_acc:.2f}%")
print(f"Device-aware TNN final accuracy  : {dat_final_acc:.2f}%")
print(f"Difference DAT - Digital         : {dat_final_acc - digital_final_acc:+.2f} pp")

'''
========================================================================
DIGITAL vs DEVICE-AWARE TRAINING SUMMARY
========================================================================
Digital TNN final accuracy       : 91.53%
Device-aware TNN final accuracy  : 93.68%
Difference DAT - Digital         : +2.15 pp

'''






# ============================================================
# 10 PROFESSIONAL COMPARISON FIGURES
# Digital TNN vs Device-Aware TNN
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

COMPARE_FIG_DIR =  "comparison_figures"

plt.rcParams.update({
    "figure.dpi": 160,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
})


# ============================================================
# Helper functions
# ============================================================

def get_preds_logits(model, loader):
    model.eval()
    ys, preds, logits = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            out = model(x)
            pred = out.argmax(dim=1)

            ys.append(y.cpu())
            preds.append(pred.cpu())
            logits.append(out.cpu())

    return (
        torch.cat(ys).numpy(),
        torch.cat(preds).numpy(),
        torch.cat(logits).numpy()
    )


def get_first_batch_images(loader, n=16):
    x, y = next(iter(loader))
    return x[:n].cpu(), y[:n].cpu().numpy()


def get_layer_weights(model):
    weights = {}
    for name, module in model.named_modules():
        if hasattr(module, "weight") and module.weight is not None:
            try:
                W = module.weight.detach().cpu().numpy()
                if W.ndim == 2:
                    weights[name] = W
            except Exception:
                pass
    return weights


def confusion_matrix_np(y_true, y_pred, n_classes=10):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def save_show():
    plt.tight_layout()
    #plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()
















# ============================================================
# Collect data
# ============================================================



from pathlib import Path
import sys

#from 04_Load_ANN_TNN.py import load_ann , load_tnn_ternary , print_model_summary
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
import importlib
_load = importlib.import_module(f"04_Load_ANN_TNN")
load_ann = _load.load_ann
load_tnn_ternary = _load.load_tnn_ternary
print_model_summary = _load.print_model_summary

TNN_CHECKPOINT = choose_checkpoint("tnn")


net_tnn_loaded, tnn_path, tnn_meta = load_tnn_ternary(TNN_CHECKPOINT)
print_model_summary(net_tnn_loaded, "TNN (ternary -1/0/+1)", tnn_path, tnn_meta)

'''
TNN (ternary -1/0/+1)
  checkpoint: TERNARY_ONLY_mnist_tw0.7_th0.05_seed42_20260522.pth
  parameters: 235,146
  saved test accuracy: 93.78%
  layers: fc1 (256, 784), fc2 (128, 256), fc3 (10, 128)
  
'''


#if only 26 epoch you can use this
#y_true_dig, pred_dig, logits_dig = get_preds_logits(digital_tnn, test_loader)


#if you want to use 100 
y_true_dig, pred_dig, logits_dig = get_preds_logits(net_tnn_loaded, test_loader)


y_true_dat, pred_dat, logits_dat = get_preds_logits(device_aware_tnn, test_loader)

acc_dig = 100 * np.mean(pred_dig == y_true_dig)
acc_dat = 100 * np.mean(pred_dat == y_true_dat)

W_dig = get_layer_weights(digital_tnn)
W_dat = get_layer_weights(device_aware_tnn)



########################################################################
#################### Only for 26 epoch ################################
########################################################################

# ============================================================
# FIGURE 1 — Training loss comparison
# ============================================================

plt.figure(figsize=(8.5, 5.2))
plt.plot(digital_loss_history, linewidth=2.2, label="Digital TNN")
plt.plot(dat_loss_history, linewidth=2.2, label="Device-aware TNN")
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("Training Loss: Digital vs Device-Aware TNN")
plt.grid(alpha=0.3)
plt.legend(frameon=False)
save_show()


# ============================================================
# FIGURE 2 — Accuracy trajectory
# ============================================================

plt.figure(figsize=(8.5, 5.2))
plt.plot(digital_acc_history, linewidth=2.2, marker="o", markersize=3, label="Digital TNN")
plt.plot(dat_acc_history, linewidth=2.2, marker="s", markersize=3, label="Device-aware TNN")
plt.xlabel("Epoch")
plt.ylabel("Test accuracy (%)")
plt.title("Accuracy Recovery during Device-Aware Training")
plt.grid(alpha=0.3)
plt.legend(frameon=False)
save_show()

# ============================================================
# FIGURE 3 — Final accuracy bar plot
# ============================================================

plt.figure(figsize=(6.5, 5))
bars = plt.bar(
    ["Digital TNN", "Device-aware TNN"],
    [acc_dig, acc_dat],
    edgecolor="black",
    linewidth=0.9
)
for bar, acc in zip(bars, [acc_dig, acc_dat]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{acc:.2f}%",
        ha="center",
        fontsize=11
    )
plt.ylabel("MNIST test accuracy (%)")
plt.title("Final Accuracy Comparison")
plt.ylim(0, 100)
plt.grid(axis="y", alpha=0.3)
save_show()

########################################################################
#################### Only for 26 epoch ################################
########################################################################





# ============================================================
# FIGURE 4 — Prediction examples
# ============================================================

x_ex, y_ex = get_first_batch_images(test_loader, n=16)

plt.figure(figsize=(9, 9))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(x_ex[i].view(28, 28), cmap="gray")
    color = "green" if pred_dat[i] == y_ex[i] else "red"
    plt.title(
        f"T:{y_ex[i]} | D:{pred_dig[i]} | DAT:{pred_dat[i]}",
        fontsize=9,
        color=color
    )
    plt.axis("off")

plt.suptitle("Prediction Examples: True / Digital / Device-Aware", fontsize=14)
save_show()

# ============================================================
# FIGURE 5 — Confusion matrix difference
# ============================================================

cm_dig = confusion_matrix_np(y_true_dig, pred_dig)
cm_dat = confusion_matrix_np(y_true_dat, pred_dat)
cm_diff = cm_dat - cm_dig

plt.figure(figsize=(7, 6))
im = plt.imshow(cm_diff, cmap="RdBu_r", aspect="auto")
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix Difference\nDevice-Aware − Digital")
plt.xticks(range(10))
plt.yticks(range(10))

for i in range(10):
    for j in range(10):
        val = cm_diff[i, j]
        if val != 0:
            plt.text(j, i, str(val), ha="center", va="center", fontsize=8)

save_show()

# ============================================================
# FIGURE 6 — Logit confidence distribution
# ============================================================

conf_dig = np.max(torch.softmax(torch.tensor(logits_dig), dim=1).numpy(), axis=1)
conf_dat = np.max(torch.softmax(torch.tensor(logits_dat), dim=1).numpy(), axis=1)

plt.figure(figsize=(8.5, 5.2))
plt.hist(conf_dig, bins=50, alpha=0.6, label="Digital TNN")
plt.hist(conf_dat, bins=50, alpha=0.6, label="Device-aware TNN")
plt.xlabel("Maximum softmax confidence")
plt.ylabel("Count")
plt.title("Prediction Confidence Distribution")
plt.grid(axis="y", alpha=0.3)
plt.legend(frameon=False)
save_show()

# ============================================================
# FIGURE 7 — Weight histograms per layer
# ============================================================

layer_names = list(W_dig.keys())[:3]

fig, axes = plt.subplots(len(layer_names), 1, figsize=(8.5, 4 * len(layer_names)))

if len(layer_names) == 1:
    axes = [axes]

for ax, name in zip(axes, layer_names):
    ax.hist(W_dig[name].ravel(), bins=60, alpha=0.55, label="Digital")
    ax.hist(W_dat[name].ravel(), bins=60, alpha=0.55, label="Device-aware")
    ax.set_title(f"Weight Distribution — {name}")
    ax.set_xlabel("Weight value")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)

save_show()

# ============================================================
# FIGURE 8 — Weight heatmaps for 3 layers
# ============================================================

fig, axes = plt.subplots(3, 2, figsize=(12, 12))

for row, name in enumerate(layer_names):
    W1 = W_dig[name]
    W2 = W_dat[name]

    vmax = np.percentile(np.abs(np.concatenate([W1.ravel(), W2.ravel()])), 99)
    vmax = max(vmax, 1e-6)

    im0 = axes[row, 0].imshow(W1, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[row, 0].set_title(f"Digital — {name}")
    axes[row, 0].set_xlabel("Input index")
    axes[row, 0].set_ylabel("Output index")

    im1 = axes[row, 1].imshow(W2, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[row, 1].set_title(f"Device-aware — {name}")
    axes[row, 1].set_xlabel("Input index")
    axes[row, 1].set_ylabel("Output index")

fig.colorbar(im1, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
plt.suptitle("Weight Heatmaps: Digital vs Device-Aware", fontsize=15)
plt.show()


# ============================================================
# FIGURE 9 — Weight change heatmaps
# ============================================================

fig, axes = plt.subplots(1, len(layer_names), figsize=(5 * len(layer_names), 4.5))

if len(layer_names) == 1:
    axes = [axes]

for ax, name in zip(axes, layer_names):
    delta = W_dat[name] - W_dig[name]
    vmax = np.percentile(np.abs(delta), 99)
    vmax = max(vmax, 1e-6)

    im = ax.imshow(delta, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(f"ΔW Device-aware − Digital\n{name}")
    ax.set_xlabel("Input index")
    ax.set_ylabel("Output index")
    plt.colorbar(im, ax=ax, fraction=0.046)

save_show()

# ============================================================
# FIGURE 10 — Per-layer weight-change magnitude
# ============================================================

summary_rows = []

for name in layer_names:
    delta = W_dat[name] - W_dig[name]
    summary_rows.append({
        "Layer": name,
        "Mean |ΔW|": np.mean(np.abs(delta)),
        "Max |ΔW|": np.max(np.abs(delta)),
        "RMS ΔW": np.sqrt(np.mean(delta ** 2)),
    })

weight_change_df = pd.DataFrame(summary_rows)
print("\n================ WEIGHT CHANGE SUMMARY ================")
print(weight_change_df)

plt.figure(figsize=(8.5, 5.2))

x = np.arange(len(weight_change_df))
width = 0.25

plt.bar(x - width, weight_change_df["Mean |ΔW|"], width, label="Mean |ΔW|")
plt.bar(x, weight_change_df["RMS ΔW"], width, label="RMS ΔW")
plt.bar(x + width, weight_change_df["Max |ΔW|"], width, label="Max |ΔW|")

plt.xticks(x, weight_change_df["Layer"])
plt.ylabel("Weight-change magnitude")
plt.title("Per-Layer Weight Adaptation from Device-Aware Training")
plt.grid(axis="y", alpha=0.3)
plt.legend(frameon=False)
save_show()








# ============================================================
# FINAL THESIS FIGURE
# Ultra high-DPI academic figure
# ============================================================

import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Final accuracies
# ------------------------------------------------------------

labels = [
    "Digital TNN",
    "Analog TNN",
    "Device-Aware\nAnalog TNN"
]

accuracies = [
    93.78,
    81.14,
    93.68
]

# ------------------------------------------------------------
# Elegant academic palette
# ------------------------------------------------------------

colors = [
    "#1F3A5F",   # deep academic blue
    "#7A7A7A",   # neutral gray
    "#B08D57"    # muted gold
]

# ------------------------------------------------------------
# Create figure
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8.8, 5.8),
    dpi=600
)

bars = ax.bar(
    labels,
    accuracies,
    width=0.56,
    color=colors,
    edgecolor="black",
    linewidth=1.1,
    zorder=3
)

# ------------------------------------------------------------
# Accuracy labels
# ------------------------------------------------------------

for bar, acc in zip(bars, accuracies):

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        acc + 0.9,
        f"{acc:.2f}%",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold"
    )

# ------------------------------------------------------------
# Axis styling
# ------------------------------------------------------------

ax.set_ylim(0, 100)

ax.set_ylabel(
    "MNIST Test Accuracy (%)",
    fontsize=13
)

ax.set_title(
    "Performance Recovery using Device-Aware Training\n"
    "for APM_SINW_SBFET Ternary Neural Networks",
    fontsize=10,
    pad=18
)

# ------------------------------------------------------------
# Grid styling
# ------------------------------------------------------------

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.28,
    zorder=0
)

# ------------------------------------------------------------
# Clean academic spines
# ------------------------------------------------------------

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_linewidth(1.0)
ax.spines["bottom"].set_linewidth(1.0)

# ------------------------------------------------------------
# Tick styling
# ------------------------------------------------------------

ax.tick_params(
    axis="x",
    labelsize=12
)

ax.tick_params(
    axis="y",
    labelsize=11
)

# ------------------------------------------------------------
# Recovery annotation
# ------------------------------------------------------------

recovery = accuracies[2] - accuracies[1]

ax.text(
    1,
    90,
    f"Accuracy recovery: +{recovery:.2f} pp",
    ha="center",
    fontsize=11,
    fontweight="bold",
    color="#444444"
)

# ------------------------------------------------------------
# Export ultra high-quality figure
# ------------------------------------------------------------

plt.tight_layout()

plt.savefig(
    "final_device_aware_comparison.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()
























# ============================================================
# 8. Reload saved checkpoints — inspect weight distributions
# ============================================================

TERNARY_BINS = [-1.5, -0.5, 0.5, 1.5]
TERNARY_TICKS = [-1, 0, 1]
LAYER_NAMES = ("fc1", "fc2", "fc3")


def extract_stored_weights(state_dict):
    """Read fc1/fc2/fc3 weight tensors directly from a checkpoint state_dict."""
    weights = {}
    for name in LAYER_NAMES:
        key = f"{name}.weight"
        if key in state_dict:
            w = state_dict[key].detach().cpu().numpy()
            if w.ndim == 2:
                weights[name] = w
    return weights


def to_used_ternary(w):
    """Map stored weights to {-1, 0, +1} as used in forward pass."""
    return np.where(
        w > THRESHOLD,
        1.0,
        np.where(w < -THRESHOLD, -1.0, 0.0),
    )


def is_strictly_ternary(w, tol=1e-6):
    rounded = np.round(w)
    return np.all(np.isin(rounded, [-1.0, 0.0, 1.0])) and np.allclose(w, rounded, atol=tol)


def print_weight_report(weights, model_label):
    print(f"\n{model_label}")
    print("-" * 60)
    for name, w in weights.items():
        w_used = to_used_ternary(w)
        uniq = np.unique(np.round(w.ravel(), 6))
        print(f"  {name}  shape={w.shape}")
        print(f"    stored: min={w.min():+.4f}  max={w.max():+.4f}  "
              f"mean={w.mean():+.4f}  std={w.std():.4f}")
        print(f"    unique stored values: {uniq.size:,}")
        print(f"    strictly ternary in checkpoint: {is_strictly_ternary(w)}")
        print("    used ternary (-1 / 0 / +1):")
        for val in (-1.0, 0.0, 1.0):
            n = int((w_used == val).sum())
            print(f"      {val:+.0f}: {n:,} ({100.0 * n / w_used.size:.2f}%)")


def plot_checkpoint_weights(weights, model_label):
    """Per-layer stored-weight histogram + used-ternary histogram for one checkpoint."""
    n_layers = len(weights)
    fig, axes = plt.subplots(n_layers, 2, figsize=(12, 3.8 * n_layers))
    if n_layers == 1:
        axes = np.array([axes])

    for row, (name, w) in enumerate(weights.items()):
        w_flat = w.ravel()
        w_used = to_used_ternary(w)

        axes[row, 0].hist(
            w_flat, bins=80, color="#2C5282",
            edgecolor="white", alpha=0.85,
        )
        axes[row, 0].axvline(-THRESHOLD, color="red", linestyle="--", linewidth=1, label=f"±{THRESHOLD}")
        axes[row, 0].axvline(+THRESHOLD, color="red", linestyle="--", linewidth=1)
        axes[row, 0].axvline(0, color="black", linestyle=":", linewidth=0.8)
        axes[row, 0].set_title(f"{name} — stored weights (checkpoint)")
        axes[row, 0].set_xlabel("weight value")
        axes[row, 0].set_ylabel("count")
        axes[row, 0].grid(axis="y", alpha=0.3)
        axes[row, 0].legend(fontsize=8)

        axes[row, 1].hist(
            w_used.ravel(), bins=TERNARY_BINS, color="#1B7F3B",
            edgecolor="black", alpha=0.9, align="mid",
        )
        axes[row, 1].set_xticks(TERNARY_TICKS)
        axes[row, 1].set_title(f"{name} — used ternary weights")
        axes[row, 1].set_xlabel("weight value")
        axes[row, 1].set_ylabel("count")
        axes[row, 1].grid(axis="y", alpha=0.3)

    plt.suptitle(f"{model_label} — weight distribution", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


# --- digital_tnn.pt ---
#26epochs
digital_ckpt = torch.load(choose_checkpoint("digital_tnn"), map_location="cpu")
W_digital = extract_stored_weights(digital_ckpt["state_dict"])

print("\n" + "=" * 72)
print("CHECKPOINT WEIGHT INSPECTION")
print("=" * 72)
print(f"digital_tnn.pt       — final accuracy: {digital_ckpt['final_accuracy']:.2f}%")
print_weight_report(W_digital, "digital_tnn.pt")
plot_checkpoint_weights(W_digital, "digital_tnn.pt")





#100 epochs
digital_ckpt = torch.load(TNN_CHECKPOINT, map_location="cpu")
W_digital = extract_stored_weights(digital_ckpt["state_dict"])
print("\n" + "=" * 72)
print("CHECKPOINT WEIGHT INSPECTION")
print("=" * 72)
print_weight_report(W_digital, "digital_tnn.pt")
plot_checkpoint_weights(W_digital, "digital_tnn.pt")






# --- device_aware_tnn.pt ---
dat_ckpt = torch.load(choose_checkpoint("device_aware_tnn"), map_location="cpu")
W_dat = extract_stored_weights(dat_ckpt["state_dict"])

print(f"\ndevice_aware_tnn.pt  — final accuracy: {dat_ckpt['final_accuracy']:.2f}%")
print_weight_report(W_dat, "device_aware_tnn.pt")
plot_checkpoint_weights(W_dat, "device_aware_tnn.pt")





