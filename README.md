# Device–Algorithm Co-Design of Ternary Weight Encoding for Two-State Silicon Nanowire Memristive Crossbars

**Ali Pilehvar Meibody**<sup>1</sup>, **Junrui Chen**<sup>1,*</sup>, **Aferdita Xhameni**<sup>2,3</sup>, **Antonio Lombardo**<sup>2,3</sup>, and **Sandro Carrara**<sup>1</sup>

<sup>1</sup> Bio/CMOS Interfaces Laboratory, École polytechnique fédérale de Lausanne (EPFL), Neuchâtel, Switzerland  
<sup>2</sup> London Centre for Nanotechnology, University College London, London WC1H 0AH, United Kingdom  
<sup>3</sup> Department of Electronic and Electrical Engineering, University College London, London WC1E 7JE, United Kingdom  

<sup>*</sup> Correspondence: [junrui.chen@epfl.ch](mailto:junrui.chen@epfl.ch)

This repository is the code and measured device data for the manuscript (September 2026). It trains a ternary MNIST network whose weights lie in $\{-1, 0, +1\}$, maps those weights onto a differential crossbar that uses only the two measured states of a silicon-nanowire memristor, and evaluates that mapping in an experimentally informed [CrossSim](https://github.com/sandialabs/cross-sim) model.

## Result

The silicon-nanowire devices provide a high-resistance state and a low-resistance state. Ternary synapses are not a third conductance. Each weight in $\{-1, 0, +1\}$ is a differential pair of those two states, so a layer needs two crossbars rather than a multilevel cell.

On MNIST, with a 784–256–128–10 network:

| Condition | Test accuracy | Where it is produced |
|---|---:|---|
| Digital ternary network | 93.78% | `03_Base_ANN_TNN.py` |
| Programming variability only | 92.56% | `08_Real_TNN_Cross_S.py` |
| Programming variability, drift, 6-bit DAC / 4-bit ADC, and $0.01\,\Omega$ interconnect | 81.14% | `09_Final_TNN_Dev_ADC_DAC_Paraisitic.py` |
| Same hardware model after device-aware training | 93.68% | `10_Device_Aware_Training.py` |

Programming variability lowers accuracy only from 93.78% to 92.56%. The selected 6-bit DAC / 4-bit ADC is the dominant loss, down to 81.14%. Device-aware training brings the hardware-mapped network back to 93.68%.

The full-precision digital network reaches 98.00% (`01_First_ANN.py`, `03_Base_ANN_TNN.py`). Against an 8-bit bit-sliced balanced implementation of that network, the ternary mapping uses 2 crossbar arrays per layer instead of 16 (`07_Real_Cross_Sim_TNN_vs_ANN.py`): eight slices, each with a positive and a negative array, versus one differential pair. That is an eightfold reduction in arrays and devices, while most of the full-precision accuracy remains.

At the gigaohm resistance of these devices, interconnect drop is negligible. In `09_Final_TNN_Dev_ADC_DAC_Paraisitic.py`, accuracy stays at 92.56% from $R_p = 0$ through $10\,\Omega$ when only parasitics are added to the programmed array. Peripheral quantization, not wire resistance, sets the hardware accuracy.

## What is implemented

All comparisons use the same multilayer perceptron: flattened MNIST digits, two ReLU hidden layers (256 and 128 units), and 10 logits. Training uses seed 42, batch size 16, and Adam at $10^{-4}$.

- **Full-precision ANN.** Float weights. The hardware counterpart is an 8-bit bit-sliced balanced crossbar.
- **Binary network.** Forward weights in $\{-1, +1\}$. This is the two-state encoding with no zero synapse (`02b_BNN_optimization.py`).
- **Ternary network.** A float latent weight is trained with a smooth threshold-window surrogate (width 0.7). The exported inference weights are exactly $\{-1, 0, +1\}$, quantized at threshold 0.05 (`03_Base_ANN_TNN.py`).

`05_Device_Modeling.py` builds the device model from measured data. Five Ron–Roff repeats on device D3, in the stable window $t < 1\,\mathrm{s}$, set the target ON and OFF currents at $V_\mathrm{read} = +1\,\mathrm{V}$, and therefore $R_\mathrm{on}$ and $R_\mathrm{off}$. Fifty I–V sweeps in `APM_Datafinal.xlsx` give the programming error about those targets, together with $R_\min$, $R_\max$, and the relative spread of a programmed state. The resulting `APM_SINW_SBFET` model is what CrossSim uses from script 07 onward. `06_Ideal_Cross_Sim.py` first checks that an ideal CrossSim core reproduces the PyTorch ternary accuracy (93.78%) before those non-idealities are enabled.

## From a paper result to a script

Run the scripts in this order. Each one writes checkpoints or constants that the next one reads.

| Step | Script | What it corresponds to in the paper |
|---|---|---|
| 1 | `01_First_ANN.py` | Full-precision digital baseline (98.00%) |
| 2 | `02_TNN_Optimization.py` | Choice of the ternary backprop surrogate. The retained setting is the smooth threshold window, width 0.7 |
| 3 | `02b_BNN_optimization.py` | Binary weight set $\{-1, +1\}$ |
| 4 | `03_Base_ANN_TNN.py` | Digital ANN, trained TNN, and ternary-only weights (93.78%) |
| 5 | `04_Load_ANN_TNN.py`, `04b_Load_ANN_BNN_TNN.py` | Reload and compare the saved ANN, BNN, and TNN |
| 6 | `05_Device_Modeling.py` | HRS/LRS targets, programming error, drift, $R_\min$, $R_\max$ |
| 7 | `06_Ideal_Cross_Sim.py` | Ideal CrossSim matches PyTorch (bitsliced ANN and balanced TNN) |
| 8 | `07_Real_Cross_Sim_TNN_vs_ANN.py` | Programming error on ANN and TNN; array count 16 versus 2 per layer |
| 9 | `08_Real_TNN_Cross_S.py` | TNN with programming variability (92.56%) and with drift |
| 10 | `09_Final_TNN_Dev_ADC_DAC_Paraisitic.py` | 6-bit DAC / 4-bit ADC and interconnect resistance (81.14%) |
| 11 | `10_Device_Aware_Training.py` | Device-aware recovery to 93.68% |

`paths.py` only locates folders and checkpoints. It is not an experimental step.

## Layout

Paths are relative to this repository.

```text
Pth_Models/       trained weights (.pth, .pt)
Device_Data/      Ron–Roff CSVs and APM_Datafinal.xlsx
data/             MNIST (downloaded on first run)
figures/          saved plots
cross-sim/        CrossSim checkout, required from script 06
```

`Device_Data/` holds `*_Ron-off*.csv` for devices D1–D3 (five repeats each) and `APM_Datafinal.xlsx` (100 columns, 50 sweeps). Script 05 reads both.

Override a location only when the files live elsewhere:

| Variable | Default |
|---|---|
| `PTH_DIR` | `Pth_Models/` |
| `DEVICE_DATA_DIR` | `Device_Data/` |
| `MNIST_DIR` | `data/` |
| `CROSS_SIM_DIR` | `cross-sim/`, then `../cross-sim` |

## Running

The scripts need Python with PyTorch, torchvision, NumPy, Matplotlib, pandas, and openpyxl. Scripts 06–10 also need CrossSim:

```bash
git clone https://github.com/sandialabs/cross-sim.git cross-sim
python 03_Base_ANN_TNN.py
```

MNIST is stored in `data/` the first time a script loads it.

Scripts that load a network list the matching files in `Pth_Models/`. One match is used. If several match, the script prints a numbered list; Enter keeps the newest. To skip the prompt, set `CHECKPOINT_ANN`, `CHECKPOINT_BNN`, `CHECKPOINT_TNN`, `CHECKPOINT_TNN_TRAIN`, `CHECKPOINT_DIGITAL_TNN`, or `CHECKPOINT_DEVICE_AWARE_TNN` to a file path.

The ternary-only checkpoint used for the 93.78% result is `Pth_Models/TERNARY_ONLY_mnist_tw0.7_th0.05_seed42_20260908.pth`. The full-precision ANN is `Pth_Models/BASE_ANN_mnist_lr0.0001_ep100_seed42_20260908.pth`. Device-aware training writes `Pth_Models/Device_aware/{date}_digital_tnn.pt` and `{date}_device_aware_tnn.pt`.

## License

MIT. See [LICENSE](LICENSE). Copyright (c) 2026 Ali Pilehvar Meibody.
