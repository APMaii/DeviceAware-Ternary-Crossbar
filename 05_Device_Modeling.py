'''
In The Name of God

Ali Pilehvar Meibody

Last Update : 24 May 2026


Here we just want to model programming error and also Drift error and we 
add new model into cross sim to can be accessible in paramters for using in
07_Real_Cross_Sim.py





section1 : Programming error mdoeling 
First in device_data and drift data that we have for device 3 , we have the current
of On and   Off state for 5 repttion(at t<1 stable) so we can get mean of that and this is our 
TARGET ON current and TARGET OFF Current . and because we have our read voltage 
we can calculate R_on and R_of


Then we go for I-V data from APM_Datafinal.xlsx that we have 100 columns (50 sweeps) 
so from each sweep we see on, ofr current at Vread= +1 V
so for each sweeep with np.gradient(V) we seperate forward and backward 
then we get interpolation at 1v . the bigger is On, the smaller is OFF.
then we can get programming error 

On_error = Ion_actual - TARGET_ON
OFF_Error = Ioff_actual - TARGET_OFF
 


R min , Rmax , Rmin comes from here
and also  we got sigma relative means that when you switch on on or off
you didn't get exact number, you get with some standard deviation






This code (06_b) is only on all d1,d2,d3 and ... to have also device to device change

'''





#################################################################
#################################################################
#################################################################
#################################################################
'''                Programming error modeling              '''
#################################################################
#################################################################
#################################################################
#################################################################

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

from paths import DEVICE_DATA_DIR



# ============================================================
# SECTION 0 — IMPORTS AND PATHS
# ============================================================

DRIFT_DIR = str(DEVICE_DATA_DIR)
IV_FILE = str(DEVICE_DATA_DIR / "APM_Datafinal.xlsx")

drift_files = sorted(glob.glob(os.path.join(DRIFT_DIR, "*_Ron-off*.csv")))


VREAD = 1.0
INITIAL_WINDOW_SEC = 1.0


# ============================================================
# SECTION 1 — LOAD ONLY D3 RON-OFF DRIFT FILES
# ============================================================


# Publication-style defaults (thesis figures)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "axes.titleweight": "normal",
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#cccccc",
    "legend.fontsize": 12,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 120,
    "savefig.dpi": 300,
})

# Colorblind-friendly palette (Okabe–Ito)
C_ON = "#03a3fc"
C_OFF = "#fc7103"
C_TARGET_ON = "#020e6b"
C_TARGET_OFF = "#854900"
C_NEUTRAL = "#333333"
C_IV = "#2E5090"
C_SERIES = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
HIST_KW = dict(alpha=0.72, edgecolor="white", linewidth=0.6)



print("Number of D3 drift files:", len(drift_files))
for f in drift_files:
    print(os.path.basename(f))

if len(drift_files) == 0:
    raise FileNotFoundError(f"No *_Ron-off*.csv files in {DRIFT_DIR}")


def parse_drift_filename(path):
    name = os.path.basename(path).replace(".csv", "")
    device = name.split("_")[0]
    repeat = int(name.split("_")[1].replace("Ron-off", ""))
    return device, repeat


drift_initial_rows = []

for path in drift_files:
    df = pd.read_csv(path)
    device, repeat = parse_drift_filename(path)

    off_initial = df[df["Time_off"] <= INITIAL_WINDOW_SEC]["ID_off"].abs().median()
    on_initial = df[df["Time_on"] <= INITIAL_WINDOW_SEC]["ID_on"].abs().median()

    drift_initial_rows.append({
        "device": device,
        "repeat": repeat,
        "Ioff_A": off_initial,
        "Ion_A": on_initial,
        "Ioff_nA": off_initial * 1e9,
        "Ion_nA": on_initial * 1e9,
        "Roff_ohm": VREAD / off_initial,
        "Ron_ohm": VREAD / on_initial,
    })

drift_initial = pd.DataFrame(drift_initial_rows)

print("\nInitial ON/OFF values from D3 fixed-read Ron-off data:")
print(drift_initial)

'''
Number of D3 drift files: 15
D1_Ron-off1.csv
D1_Ron-off2.csv
D1_Ron-off3.csv
D1_Ron-off4.csv
D1_Ron-off5.csv
D2_Ron-off1.csv
D2_Ron-off2.csv
D2_Ron-off3.csv
D2_Ron-off4.csv
D2_Ron-off5.csv
D3_Ron-off1.csv
D3_Ron-off2.csv
D3_Ron-off3.csv
D3_Ron-off4.csv
D3_Ron-off5.csv

Initial ON/OFF values from D3 fixed-read Ron-off data:
   device  repeat        Ioff_A  ...    Ion_nA      Roff_ohm       Ron_ohm
0      D1       1  2.259340e-10  ...  0.697935  4.426071e+09  1.432798e+09
1      D1       2  2.270770e-10  ...  0.697831  4.403793e+09  1.433012e+09
2      D1       3  2.326250e-10  ...  0.712429  4.298764e+09  1.403649e+09
3      D1       4  2.417110e-10  ...  0.711733  4.137172e+09  1.405021e+09
4      D1       5  2.224810e-10  ...  0.691747  4.494766e+09  1.445615e+09
5      D2       1  2.243610e-10  ...  0.719925  4.457103e+09  1.389034e+09
6      D2       2  2.118390e-10  ...  0.714477  4.720566e+09  1.399625e+09
7      D2       3  2.306330e-10  ...  0.708603  4.335893e+09  1.411227e+09
8      D2       4  2.135180e-10  ...  0.717113  4.683446e+09  1.394480e+09
9      D2       5  2.165660e-10  ...  0.703803  4.617530e+09  1.420852e+09
10     D3       1  2.446770e-10  ...  0.682534  4.087021e+09  1.465128e+09
11     D3       2  2.040460e-10  ...  0.733798  4.900856e+09  1.362773e+09
12     D3       3  2.614750e-10  ...  0.746359  3.824457e+09  1.339838e+09
13     D3       4  1.903240e-10  ...  0.673416  5.254198e+09  1.484966e+09
14     D3       5  2.393140e-10  ...  0.723877  4.178611e+09  1.381450e+09

[15 rows x 8 columns]

'''


# ============================================================
#  TARGET ON/OFF STATES FROM D3
# ============================================================

TARGET_ON_A = drift_initial["Ion_A"].mean()
TARGET_OFF_A = drift_initial["Ioff_A"].mean()

TARGET_ON_nA = TARGET_ON_A * 1e9
TARGET_OFF_nA = TARGET_OFF_A * 1e9

R_ON = VREAD / TARGET_ON_A
R_OFF = VREAD / TARGET_OFF_A

print("\n================ TARGET STATES FROM D3 ================")
print(f"Target ON current  = {TARGET_ON_nA:.6f} nA")
print(f"Target OFF current = {TARGET_OFF_nA:.6f} nA")
print(f"R_ON        = {R_ON:.6e} ohm")
print(f"R_OFF     = {R_OFF:.6e} ohm")
print(f"ON/OFF current ratio = {TARGET_ON_A / TARGET_OFF_A:.3f}")
print(f"R_OFF/R_ON ratio      = {R_OFF / R_ON:.3f}")
'''
================ TARGET STATES FROM D3 ================
Target ON current  = 0.709039 nA
Target OFF current = 0.225772 nA
R_ON        = 1.410360e+09 ohm
R_OFF     = 4.429246e+09 ohm
ON/OFF current ratio = 3.141
R_OFF/R_ON ratio      = 3.141

'''


plt.figure(figsize=(7, 5))
plt.hist(drift_initial["Ion_nA"], bins=5, color=C_ON, label="Initial ON", **HIST_KW)
plt.hist(drift_initial["Ioff_nA"], bins=5, color=C_OFF, label="Initial OFF", **HIST_KW)
plt.axvline(TARGET_ON_nA, color=C_TARGET_ON, linestyle="--", linewidth=1.8, label="Target ON mean")
plt.axvline(TARGET_OFF_nA, color=C_TARGET_OFF, linestyle="--", linewidth=1.8, label="Target OFF mean")
plt.xlabel("Initial current at Vread = 1V (nA)")
plt.ylabel("Count")
plt.title("Target ON/OFF States from Ron-off Data")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
#  LOAD I-V HYSTERESIS DATA
# ============================================================

if not os.path.isfile(IV_FILE):
    raise FileNotFoundError(
        f"Missing I-V workbook: {IV_FILE}\n"
        "Put APM_Datafinal.xlsx in Device_Data/, or set DEVICE_DATA_DIR."
    )

raw = pd.read_excel(IV_FILE, header=None)

num_cols = raw.shape[1]
num_sweeps = num_cols // 2

print("\nRows:", raw.shape[0])
print("Columns:", raw.shape[1])
print("Number of I-V sweeps:", num_sweeps)

sweeps = []

for i in range(num_sweeps):
    V = raw.iloc[:, 2*i].to_numpy(dtype=float)
    I = raw.iloc[:, 2*i + 1].to_numpy(dtype=float)

    sweeps.append({
        "sweep": i + 1,
        "V": V,
        "I": I,
        "absI": np.abs(I)
    })

# Column order in the Excel file: sweep 1 is the first pair, sweep 50 the last.
# Light = earlier sweep, dark = later sweep.
sweep_cmap = LinearSegmentedColormap.from_list(
    "sweep_order", ["#D5D8DE", "#1B3A6B"]
)
sweep_norm = Normalize(vmin=1, vmax=num_sweeps)

fig, ax = plt.subplots(figsize=(9, 6))
for s in sweeps:
    ax.plot(
        s["V"],
        s["I"] * 1e9,
        color=sweep_cmap(sweep_norm(s["sweep"])),
        linewidth=0.9,
    )

sweep_guide = ScalarMappable(norm=sweep_norm, cmap=sweep_cmap)
sweep_guide.set_array([])
cbar = fig.colorbar(sweep_guide, ax=ax, pad=0.02)
cbar.set_label("Sweep number")
cbar.set_ticks([1, 10, 20, 30, 40, num_sweeps])

ax.set_xlabel("Voltage (V)")
ax.set_ylabel("Current (nA)")
ax.set_title("APM SiNW SBFET - 50 I-V Hysteresis Sweeps")
ax.grid(True)
fig.tight_layout()
plt.show()


# ============================================================
# SECTION 4 — EXTRACT ACTUAL ON/OFF BRANCH CURRENTS AT +1V
# ============================================================
'''
when we upload I-V data we set 
'''

actual_rows = []

for s in sweeps:
    V = s["V"]
    Iabs = np.abs(s["I"])

    dV = np.gradient(V)

    forward_mask = dV > 0
    backward_mask = dV < 0

    V_forward = V[forward_mask]
    I_forward = Iabs[forward_mask]

    V_backward = V[backward_mask]
    I_backward = Iabs[backward_mask]

    idx_f = np.argsort(V_forward)
    idx_b = np.argsort(V_backward)

    I_forward_at_read = np.interp(
        VREAD,
        V_forward[idx_f],
        I_forward[idx_f]
    )

    I_backward_at_read = np.interp(
        VREAD,
        V_backward[idx_b],
        I_backward[idx_b]
    )

    I_on_actual = max(I_forward_at_read, I_backward_at_read)
    I_off_actual = min(I_forward_at_read, I_backward_at_read)

    actual_rows.append({
        "sweep": s["sweep"],
        "I_forward_A": I_forward_at_read,
        "I_backward_A": I_backward_at_read,
        "Ion_actual_A": I_on_actual,
        "Ioff_actual_A": I_off_actual,
        "Ion_actual_nA": I_on_actual * 1e9,
        "Ioff_actual_nA": I_off_actual * 1e9,
    })

actual = pd.DataFrame(actual_rows)

print("\nActual ON/OFF states extracted from hysteresis sweeps at +1V:")
print(actual.head())

'''
Actual ON/OFF states extracted from hysteresis sweeps at +1V:
   sweep   I_forward_A  ...  Ion_actual_nA  Ioff_actual_nA
0      1  1.988353e-10  ...       0.709827        0.198835
1      2  2.031484e-10  ...       0.725225        0.203148
2      3  2.024438e-10  ...       0.722709        0.202444
3      4  1.999396e-10  ...       0.713770        0.199940
4      5  1.952815e-10  ...       0.697140        0.195281

[5 rows x 7 columns]
'''

print("\nActual ON current stats:")
print(actual["Ion_actual_nA"].describe())

'''
Actual ON current stats:
count    50.000000
mean      0.716448
std       0.034374
min       0.680380
25%       0.699740
50%       0.710195
75%       0.719142
max       0.888400
Name: Ion_actual_nA, dtype: float64

'''

print("\nActual OFF current stats:")
print(actual["Ioff_actual_nA"].describe())

'''
Actual OFF current stats:
count    50.000000
mean      0.197050
std       0.012295
min       0.171820
25%       0.193035
50%       0.198416
75%       0.200990
max       0.257190
Name: Ioff_actual_nA, dtype: float64

'''


#better-------
plt.figure(figsize=(7, 5))
plt.hist(actual["Ion_actual_nA"], bins=15, color=C_ON, label="Actual ON / LRS", **HIST_KW)
plt.hist(actual["Ioff_actual_nA"], bins=15, color=C_OFF, label="Actual OFF / HRS", **HIST_KW)
plt.axvline(TARGET_ON_nA, color=C_TARGET_ON, linestyle="--", linewidth=1.8, label="Target ON ")
plt.axvline(TARGET_OFF_nA, color=C_TARGET_OFF, linestyle="--", linewidth=1.8, label="Target OFF ")
plt.xlabel("Current at +1V (nA)")
plt.ylabel("Count")
plt.title("Actual Hysteresis ON/OFF vs  Target States")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# SECTION 5 — PROGRAMMING ERROR = ACTUAL - D3 TARGET
# ============================================================

actual["ON_error_A"] = actual["Ion_actual_A"] - TARGET_ON_A
actual["OFF_error_A"] = actual["Ioff_actual_A"] - TARGET_OFF_A

actual["ON_error_nA"] = actual["ON_error_A"] * 1e9
actual["OFF_error_nA"] = actual["OFF_error_A"] * 1e9

actual["ON_error_rel"] = actual["ON_error_A"] / TARGET_ON_A
actual["OFF_error_rel"] = actual["OFF_error_A"] / TARGET_OFF_A

print("\n================ PROGRAMMING ERROR SUMMARY VS D3 TARGET ================")

print("\nON error, nA:")
print(actual["ON_error_nA"].describe())

'''
================ PROGRAMMING ERROR SUMMARY VS D3 TARGET ================

ON error, nA:
count    50.000000
mean      0.007409
std       0.034374
min      -0.028659
25%      -0.009299
50%       0.001156
75%       0.010103
max       0.179361
Name: ON_error_nA, dtype: float64

'''


print("\nOFF error, nA:")
print(actual["OFF_error_nA"].describe())

'''
OFF error, nA:
count    50.000000
mean     -0.028722
std       0.012295
min      -0.053952
25%      -0.032737
50%      -0.027356
75%      -0.024782
max       0.031418
Name: OFF_error_nA, dtype: float64

'''


print("\nON relative error:")
print(actual["ON_error_rel"].describe())

'''
ON relative error:
count    50.000000
mean      0.010450
std       0.048479
min      -0.040419
25%      -0.013115
50%       0.001631
75%       0.014249
max       0.252964
Name: ON_error_rel, dtype: float64

'''

print("\nOFF relative error:")
print(actual["OFF_error_rel"].describe())

'''
OFF relative error:
count    50.000000
mean     -0.127216
std       0.054457
min      -0.238967
25%      -0.145002
50%      -0.121166
75%      -0.109766
max       0.139158
Name: OFF_error_rel, dtype: float64

'''


ON_SIGMA_REL = actual["ON_error_rel"].std()
OFF_SIGMA_REL = actual["OFF_error_rel"].std()

ON_ERROR_MIN_REL = actual["ON_error_rel"].min()
ON_ERROR_MAX_REL = actual["ON_error_rel"].max()

OFF_ERROR_MIN_REL = actual["OFF_error_rel"].min()
OFF_ERROR_MAX_REL = actual["OFF_error_rel"].max()

print("\nRelative programming error parameters:")
print(f"ON sigma rel  = {ON_SIGMA_REL:.6f}")
print(f"OFF sigma rel = {OFF_SIGMA_REL:.6f}")
print(f"ON rel clip   = [{ON_ERROR_MIN_REL:.6f}, {ON_ERROR_MAX_REL:.6f}]")
print(f"OFF rel clip  = [{OFF_ERROR_MIN_REL:.6f}, {OFF_ERROR_MAX_REL:.6f}]")

'''
Relative programming error parameters:
ON sigma rel  = 0.048479
OFF sigma rel = 0.054457
ON rel clip   = [-0.040419, 0.252964]
OFF rel clip  = [-0.238967, 0.139158]

'''


#better***
plt.figure(figsize=(7, 5))
plt.hist(actual["ON_error_rel"] * 100, bins=15, color=C_ON, label="ON programming error", **HIST_KW)
plt.hist(actual["OFF_error_rel"] * 100, bins=15, color=C_OFF, label="OFF programming error", **HIST_KW)
plt.axvline(0, color=C_NEUTRAL, linestyle="--", linewidth=1.8)
plt.xlabel("Relative programming error (%)")
plt.ylabel("Count")
plt.title("Programming Error Distribution: Hysteresis Actual -Target")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# SECTION 5B — CONDUCTANCE ERROR VS CONDUCTANCE
# ============================================================

actual["Gon_target_nS"] = TARGET_ON_nA
actual["Goff_target_nS"] = TARGET_OFF_nA

actual["Gon_actual_nS"] = actual["Ion_actual_nA"]
actual["Goff_actual_nS"] = actual["Ioff_actual_nA"]

actual["Gon_error_nS"] = actual["Gon_actual_nS"] - actual["Gon_target_nS"]
actual["Goff_error_nS"] = actual["Goff_actual_nS"] - actual["Goff_target_nS"]

conductance_error = pd.DataFrame({
    "state": ["ON"] * len(actual) + ["OFF"] * len(actual),
    "G_target_nS": list(actual["Gon_target_nS"]) + list(actual["Goff_target_nS"]),
    "G_actual_nS": list(actual["Gon_actual_nS"]) + list(actual["Goff_actual_nS"]),
    "G_error_nS": list(actual["Gon_error_nS"]) + list(actual["Goff_error_nS"]),
})

print("\n================ CONDUCTANCE ERROR TABLE ================")
print(conductance_error.head())

print("\nConductance error summary:")
print(conductance_error.groupby("state")["G_error_nS"].describe())

'''
================ CONDUCTANCE ERROR TABLE ================
  state  G_target_nS  G_actual_nS  G_error_nS
0    ON     0.709039     0.709827    0.000789
1    ON     0.709039     0.725225    0.016186
2    ON     0.709039     0.722709    0.013671
3    ON     0.709039     0.713770    0.004731
4    ON     0.709039     0.697140   -0.011898

Conductance error summary:
       count      mean       std  ...       50%       75%       max
state                             ...                              
OFF     50.0 -0.028722  0.012295  ... -0.027356 -0.024782  0.031418
ON      50.0  0.007409  0.034374  ...  0.001156  0.010103  0.179361

[2 rows x 8 columns]

'''

plt.figure(figsize=(7, 5))
for state, color, marker in [("ON", C_ON, "o"), ("OFF", C_OFF, "s")]:
    subset = conductance_error[conductance_error["state"] == state]
    plt.scatter(
        subset["G_actual_nS"],
        subset["G_error_nS"],
        color=color,
        marker=marker,
        s=42,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.4,
        label=state
    )


#better**
plt.axhline(0, color=C_NEUTRAL, linestyle="--", linewidth=1.8)
plt.xlabel("Actual conductance at Vread = 1V (nS)")
plt.ylabel("Conductance error: G_actual - G_target (nS)")
plt.title("Conductance Error vs Conductance")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
for state, color, marker in [("ON", C_ON, "o"), ("OFF", C_OFF, "s")]:
    subset = conductance_error[conductance_error["state"] == state]
    plt.scatter(
        subset["G_target_nS"],
        subset["G_error_nS"],
        color=color,
        marker=marker,
        s=42,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.4,
        label=state
    )

plt.axhline(0, color=C_NEUTRAL, linestyle="--", linewidth=1.8)
plt.xlabel("Target conductance at Vread = 1V (nS)")
plt.ylabel("Conductance error: G_actual - G_target (nS)")
plt.title("Programming Conductance Error vs Target Conductance")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# SECTION 6 — FINAL CROSSSIM PARAMETERS FROM D3 TARGET
# ============================================================

Rmin = R_ON
Rmax = R_OFF

print("\n================ FINAL CROSSSIM DEVICE PARAMETERS FROM D3 ================")
print(f"cs_params.xbar.device.Rmin = {Rmin:.6e}   # LRS / ON")
print(f"cs_params.xbar.device.Rmax = {Rmax:.6e}   # HRS / OFF")
print("cs_params.xbar.device.cell_bits = 1")
print("cs_params.xbar.device.Vread = 1.0")

print("\nUse these programming-error parameters:")
print(f"I_ON_TARGET_NA = {TARGET_ON_nA:.12f}")
print(f"I_OFF_TARGET_NA = {TARGET_OFF_nA:.12f}")
print(f"ON_SIGMA_REL = {ON_SIGMA_REL:.12f}")
print(f"OFF_SIGMA_REL = {OFF_SIGMA_REL:.12f}")
print(f"ON_ERROR_MIN_REL = {ON_ERROR_MIN_REL:.12f}")
print(f"ON_ERROR_MAX_REL = {ON_ERROR_MAX_REL:.12f}")
print(f"OFF_ERROR_MIN_REL = {OFF_ERROR_MIN_REL:.12f}")
print(f"OFF_ERROR_MAX_REL = {OFF_ERROR_MAX_REL:.12f}")


'''
================ FINAL CROSSSIM DEVICE PARAMETERS FROM D3 ================
cs_params.xbar.device.Rmin = 1.410360e+09   # LRS / ON
cs_params.xbar.device.Rmax = 4.429246e+09   # HRS / OFF
cs_params.xbar.device.cell_bits = 1
cs_params.xbar.device.Vread = 1.0

Use these programming-error parameters:
I_ON_TARGET_NA = 0.709038666667
I_OFF_TARGET_NA = 0.225772066667
ON_SIGMA_REL = 0.048479047904
OFF_SIGMA_REL = 0.054457314136
ON_ERROR_MIN_REL = -0.040419046258
ON_ERROR_MAX_REL = 0.252964107270
OFF_ERROR_MIN_REL = -0.238966969932
OFF_ERROR_MAX_REL = 0.139157752317
'''


# ============================================================
# SECTION 7 — FINAL PROGRAMMING_ERROR FUNCTION TEMPLATE
# ============================================================

'''
def programming_error(self, input_):
    """
    APM_SINW_SBFET binary programming-error model.

    Target states are defined from D3 fixed-read Ron-off measurements
    at Vread = 1 V.

    Actual programmed states are modeled using the relative error
    distribution extracted from 50 I-V hysteresis sweeps at Vread = 1 V.

    I_actual = I_target * (1 + epsilon)

    epsilon is state-dependent and includes both:
        - measured mean relative error
        - measured standard deviation
    """

    I = self._calculate_current(input_)

    I_mid = 0.5 * (self.Imax + self.Imin)

    on_mask = I >= I_mid
    off_mask = I < I_mid

    I_programmed = xp.array(I, copy=True)

    # Units: nA
    I_ON_TARGET_NA = 0.709038666667
    I_OFF_TARGET_NA = 0.225772066667

    # Relative error mean
    ON_MEAN_REL = 0.010450
    OFF_MEAN_REL = -0.127216

    # Relative error standard deviation
    ON_SIGMA_REL = 0.048479047904
    OFF_SIGMA_REL = 0.054457314136

    # Relative error clipping range from measured data
    ON_ERROR_MIN_REL = -0.040419046258
    ON_ERROR_MAX_REL = 0.252964107270

    OFF_ERROR_MIN_REL = -0.238966969932
    OFF_ERROR_MAX_REL = 0.139157752317

    if on_mask.any():
        eps_on = xp.random.normal(
            loc=ON_MEAN_REL,
            scale=ON_SIGMA_REL,
            size=I[on_mask].shape
        )

        eps_on = xp.clip(
            eps_on,
            ON_ERROR_MIN_REL,
            ON_ERROR_MAX_REL
        )

        I_programmed[on_mask] = I_ON_TARGET_NA * (1.0 + eps_on)

    if off_mask.any():
        eps_off = xp.random.normal(
            loc=OFF_MEAN_REL,
            scale=OFF_SIGMA_REL,
            size=I[off_mask].shape
        )

        eps_off = xp.clip(
            eps_off,
            OFF_ERROR_MIN_REL,
            OFF_ERROR_MAX_REL
        )

        I_programmed[off_mask] = I_OFF_TARGET_NA * (1.0 + eps_off)

    return self._current_to_input(I_programmed)



'''






#################################################################
#################################################################
#################################################################
#################################################################
'''                Drift error modeling              '''
#################################################################
#################################################################
#################################################################
#################################################################

# ============================================================
# SECTION 8 — DRIFT MODELING (D1, D2, D3)
# Add this after the previous code
# ============================================================

from scipy.optimize import curve_fit

# Drift model:
# I(t) = I0 * (1 + t/t0)^a
T0_DRIFT = 1.0
FIT_T_MIN = 0.0
FIT_T_MAX = 100.0

FIT_T_MAX_ON = 100.0
FIT_T_MAX_OFF = 100.0

DEVICES = ["D1", "D2", "D3"]


def drift_curve_label(device, repeat):
    return f"{device}-{repeat}"


def drift_model(t, a):
    return (1.0 + t / T0_DRIFT) ** a


def on_drift_model(t, a):
    return (1.0 + t / T0_DRIFT) ** a


def off_drift_model(t, A, tau1, B, tau2):
    return (
        1.0
        + A * (1.0 - np.exp(-t / tau1))
        + B * (1.0 - np.exp(-t / tau2))
    )



def fit_single_drift_curve(t, I, state):
    t = np.asarray(t)
    I = np.asarray(I)

    fit_t_max = FIT_T_MAX_ON if state == "ON" else FIT_T_MAX_OFF

    mask = (
        np.isfinite(t)
        & np.isfinite(I)
        & (I > 0)
        & (t >= FIT_T_MIN)
        & (t <= fit_t_max)
    )

    t_fit = t[mask]
    I_fit = I[mask]

    if len(t_fit) < 10:
        return None

    initial_mask = t_fit <= INITIAL_WINDOW_SEC

    if initial_mask.sum() >= 3:
        I0 = np.median(I_fit[initial_mask])
    else:
        I0 = I_fit[0]

    I_norm = I_fit / I0

    a_guess = 0.02 if state == "ON" else -0.02
    
    if state == "ON":
        popt, _ = curve_fit(
            on_drift_model,
            t_fit,
            I_norm,
            p0=[0.02],
            maxfev=10000
        )
    
        a = popt[0]
        I_pred = I0 * on_drift_model(t_fit, a)
        
        rmse = np.sqrt(np.mean((I_fit - I_pred) ** 2))
        rmse_percent = 100 * rmse / np.mean(I_fit)
    
        return {
            "I0": I0,
            "a": a,
            "A": np.nan,
            "tau1": np.nan,
            "B": np.nan,
            "tau2": np.nan,
            "rmse": rmse,
            "rmse_percent": rmse_percent,
            "t_fit": t_fit,
            "I_fit": I_fit,
            "I_pred": I_pred,
        }
    
    
    else:
        popt, _ = curve_fit(
        off_drift_model,
        t_fit,
        I_norm,
        p0=[-0.20, 10.0, 0.08, 60.0],
        bounds=(
            [-1.0, 0.1, 0.0, 1.0],
            [0.0, 100.0, 1.0, 300.0]
        ),
        maxfev=20000
        )
        
    
        A, tau1, B, tau2 = popt
        I_pred = I0 * off_drift_model(t_fit, A, tau1, B, tau2)
        
        rmse = np.sqrt(np.mean((I_fit - I_pred) ** 2))
        rmse_percent = 100 * rmse / np.mean(I_fit)
    
        return {
            "I0": I0,
            "a": np.nan,
            "A": A,
            "tau1": tau1,
            "B": B,
            "tau2": tau2,
            "rmse": rmse,
            "rmse_percent": rmse_percent,
            "t_fit": t_fit,
            "I_fit": I_fit,
            "I_pred": I_pred,
        }
    

    


d_files = drift_files

print("\n================ DRIFT FILES (D1, D2, D3) ================")
print("Number of drift files:", len(d_files))
for f in d_files:
    print(os.path.basename(f))

if len(d_files) == 0:
    raise FileNotFoundError("No *_Ron-off*.csv files found.")

'''
================ D3 DRIFT FILES ================
Number of D3 files: 15
D1_Ron-off1.csv
D1_Ron-off2.csv
D1_Ron-off3.csv
D1_Ron-off4.csv
D1_Ron-off5.csv
D2_Ron-off1.csv
D2_Ron-off2.csv
D2_Ron-off3.csv
D2_Ron-off4.csv
D2_Ron-off5.csv
D3_Ron-off1.csv
D3_Ron-off2.csv
D3_Ron-off3.csv
D3_Ron-off4.csv
D3_Ron-off5.csv

'''


# ============================================================
# SECTION 8A — RAW DRIFT CURVES (one panel per device)
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True, sharey=True)

for ax, dev in zip(axes, DEVICES):
    for path in d_files:
        device, repeat = parse_drift_filename(path)
        if device != dev:
            continue

        df = pd.read_csv(path)
        ax.plot(
            df["Time_on"],
            df["ID_on"].abs() * 1e9,
            color=C_SERIES[(repeat - 1) % len(C_SERIES)],
            linewidth=1.6,
            alpha=0.9,
            label=drift_curve_label(device, repeat),
        )

    ax.set_title(dev,fontweight='bold')
    ax.grid(True)
    ax.legend(fontsize=8, loc="lower right")

axes[0].set_ylabel("|Ion| (nA)")
fig.supxlabel("Time (s)")
fig.suptitle("Raw ON Current Drift",fontweight='bold')
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True, sharey=True)

for ax, dev in zip(axes, DEVICES):
    for path in d_files:
        device, repeat = parse_drift_filename(path)
        if device != dev:
            continue

        df = pd.read_csv(path)
        ax.plot(
            df["Time_off"],
            df["ID_off"].abs() * 1e9,
            color=C_SERIES[(repeat - 1) % len(C_SERIES)],
            linewidth=1.6,
            alpha=0.9,
            label=drift_curve_label(device, repeat),
        )

    ax.set_title(dev,fontweight='bold')
    ax.grid(True)
    ax.legend(fontsize=8, loc="lower right")

axes[0].set_ylabel("|Ioff| (nA)")
fig.supxlabel("Time (s)")
fig.suptitle("Raw OFF Current Drift",fontweight='bold')
plt.tight_layout()
plt.show()


# ============================================================
# SECTION 8B — NORMALIZED DRIFT CURVES (one panel per device)
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True, sharey=True)

for ax, dev in zip(axes, DEVICES):
    for path in d_files:
        device, repeat = parse_drift_filename(path)
        if device != dev:
            continue

        df = pd.read_csv(path)
        Ion = df["ID_on"].abs()
        Ion0 = df[df["Time_on"] <= INITIAL_WINDOW_SEC]["ID_on"].abs().median()

        ax.plot(
            df["Time_on"],
            Ion / Ion0,
            color=C_SERIES[(repeat - 1) % len(C_SERIES)],
            linewidth=1.6,
            alpha=0.9,
            label=drift_curve_label(device, repeat),
        )

    ax.set_title(dev,fontweight='bold')
    ax.grid(True)
    ax.legend(fontsize=8, loc="lower right")

axes[0].set_ylabel("Ion(t) / Ion0")
fig.supxlabel("Time (s)")
fig.suptitle("Normalized ON Drift", fontweight='bold')
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True, sharey=True)

for ax, dev in zip(axes, DEVICES):
    for path in d_files:
        device, repeat = parse_drift_filename(path)
        if device != dev:
            continue

        df = pd.read_csv(path)
        Ioff = df["ID_off"].abs()
        Ioff0 = df[df["Time_off"] <= INITIAL_WINDOW_SEC]["ID_off"].abs().median()

        ax.plot(
            df["Time_off"],
            Ioff / Ioff0,
            color=C_SERIES[(repeat - 1) % len(C_SERIES)],
            linewidth=1.6,
            alpha=0.9,
            label=drift_curve_label(device, repeat),
        )

    ax.set_title(dev,fontweight='bold')
    ax.grid(True)
    ax.legend(fontsize=8, loc="lower right")

axes[0].set_ylabel("Ioff(t) / Ioff0")
fig.supxlabel("Time (s)")
fig.suptitle("Normalized OFF Drift",fontweight='bold')
plt.tight_layout()
plt.show()


# ============================================================
# SECTION 8C — FIT DRIFT CURVES (D1, D2, D3)
# ============================================================

d3_fit_rows = []
d3_fit_cache = {}  # key: (device, repeat, state)

for path in d_files:
    df = pd.read_csv(path)
    device, repeat = parse_drift_filename(path)

    on_fit = fit_single_drift_curve(
    df["Time_on"].values,
    df["ID_on"].abs().values,
    state="ON"
    )
    
    off_fit = fit_single_drift_curve(
        df["Time_off"].values,
        df["ID_off"].abs().values,
        state="OFF"
    )

    if on_fit is not None:
        d3_fit_rows.append({
            "device": device,
            "repeat": repeat,
            "state": "ON",
            "I0_A": on_fit["I0"],
            "I0_nA": on_fit["I0"] * 1e9,
            "a": on_fit["a"],
            "A": on_fit["A"],
            "tau1": on_fit["tau1"],
            "B": on_fit["B"],
            "tau2": on_fit["tau2"],
            "rmse_A": on_fit["rmse"],
            "rmse_percent": on_fit["rmse_percent"]
            
        })
        d3_fit_cache[(device, repeat, "ON")] = on_fit

    if off_fit is not None:
        d3_fit_rows.append({
            "device": device,
            "repeat": repeat,
            "state": "OFF",
            "I0_A": off_fit["I0"],
            "I0_nA": off_fit["I0"] * 1e9,
            "a": off_fit["a"],
            "A": off_fit["A"],
            "tau1": off_fit["tau1"],
            "B": off_fit["B"],
            "tau2": off_fit["tau2"],
            "rmse_A": off_fit["rmse"],
            "rmse_percent": off_fit["rmse_percent"]
        })
        d3_fit_cache[(device, repeat, "OFF")] = off_fit


d3_fit_results = pd.DataFrame(d3_fit_rows)

print("\n================ DRIFT FIT RESULTS (D1, D2, D3) ================")
print(d3_fit_results)
'''
================ DRIFT FIT RESULTS (D1, D2, D3) ================
   device  repeat state  ...        tau2        rmse_A  rmse_percent
0      D1       1    ON  ...         NaN  9.541754e-12      1.242318
1      D1       1   OFF  ...  300.000000  4.843266e-12      2.496562
2      D1       2    ON  ...         NaN  1.124093e-11      1.466079
3      D1       2   OFF  ...   33.256634  4.494239e-12      2.252567
4      D1       3    ON  ...         NaN  1.079048e-11      1.383765
5      D1       3   OFF  ...   36.445650  5.159112e-12      2.521148
6      D1       4    ON  ...         NaN  8.721685e-12      1.112053
7      D1       4   OFF  ...   56.836447  4.917149e-12      2.371623
8      D1       5    ON  ...         NaN  9.689274e-12      1.278290
9      D1       5   OFF  ...   30.905514  4.034217e-12      2.071998
10     D2       1    ON  ...         NaN  8.487555e-12      1.068679
11     D2       1   OFF  ...   79.954135  3.006714e-12      1.541213
12     D2       2    ON  ...         NaN  7.571015e-12      0.964436
13     D2       2   OFF  ...  300.000000  3.152635e-12      1.708130
14     D2       3    ON  ...         NaN  7.841150e-12      0.999115
15     D2       3   OFF  ...   74.756819  3.246887e-12      1.622259
16     D2       4    ON  ...         NaN  6.859097e-12      0.865682
17     D2       4   OFF  ...  300.000000  3.674730e-12      1.972770
18     D2       5    ON  ...         NaN  8.286905e-12      1.072544
19     D2       5   OFF  ...  300.000000  3.165795e-12      1.708723
20     D3       1    ON  ...         NaN  4.975179e-12      0.658005
21     D3       1   OFF  ...  300.000000  3.766971e-12      1.752248
22     D3       2    ON  ...         NaN  9.286455e-12      1.148380
23     D3       2   OFF  ...  300.000000  3.528526e-12      2.033021
24     D3       3    ON  ...         NaN  8.031092e-12      0.998424
25     D3       3   OFF  ...  300.000000  3.876240e-12      1.670059
26     D3       4    ON  ...         NaN  7.067154e-12      0.929117
27     D3       4   OFF  ...  241.815029  3.991081e-12      2.469423
28     D3       5    ON  ...         NaN  8.821913e-12      1.123905
29     D3       5   OFF  ...  300.000000  3.549270e-12      1.748329

[30 rows x 12 columns]
'''


off_fit_table = d3_fit_results[
    d3_fit_results["state"] == "OFF"
][["A", "tau1", "B", "tau2"]].dropna()


print(off_fit_table)

'''
           A       tau1         B        tau2
1  -0.279823  15.791254  0.626822  300.000000
3  -1.000000  23.326950  0.946585   33.256634
5  -1.000000  24.873327  0.962306   36.445650
7  -0.304159  12.693863  0.231730   56.836447
9  -0.996635  21.183390  0.941414   30.905514
11 -0.327217  16.405843  0.331286   79.954135
13 -0.271670  13.500160  0.704667  300.000000
15 -0.305352  15.754444  0.276956   74.756819
17 -0.294003  17.096792  0.767971  300.000000
19 -0.274824  12.745730  0.633352  300.000000
21 -0.206250  13.058626  0.383317  300.000000
23 -0.323915  15.817618  0.817728  300.000000
25 -0.214140  16.471436  0.439525  300.000000
27 -0.399050  16.531542  1.000000  241.815029
29 -0.241828  12.260364  0.401551  300.000000

'''


print("\n================ OFF FIT TABLE FOR CROSSSIM ================")
print(off_fit_table.to_string(index=False))


d3_fit_results.columns
'''
Index(['device', 'repeat', 'state', 'I0_A', 'I0_nA', 'a', 'A', 'tau1', 'B',
       'tau2', 'rmse_A', 'rmse_percent'],
      dtype='object')

'''

print("\n================ DRIFT SUMMARY BY DEVICE AND STATE ================")

d3_drift_summary = d3_fit_results.groupby(["device", "state"]).agg({
    "I0_nA": ["mean", "std", "min", "max"],
    "a": ["mean", "std", "min", "max"],
    "rmse_percent": ["mean", "std", "min", "max"]
})

print(d3_drift_summary)

print("\n================ DRIFT SUMMARY BY STATE (ALL DEVICES) ================")

d3_drift_summary_all = d3_fit_results.groupby("state").agg({
    "I0_nA": ["mean", "std", "min", "max"],
    "a": ["mean", "std", "min", "max"],
    "rmse_percent": ["mean", "std", "min", "max"]
})

print(d3_drift_summary_all)

'''
================ DRIFT SUMMARY BY DEVICE AND STATE ================
                 I0_nA                      ... rmse_percent                    
                  mean       std       min  ...          std       min       max
device state                                ...                                 
D1     OFF    0.229966  0.007511  0.222481  ...     0.185578  2.071998  2.521148
       ON     0.702335  0.009246  0.691747  ...     0.135702  1.112053  1.466079
D2     OFF    0.219383  0.007915  0.211839  ...     0.162203  1.541213  1.972770
       ON     0.712784  0.006533  0.703803  ...     0.085311  0.865682  1.072544
D3     OFF    0.227967  0.029665  0.190324  ...     0.329262  1.670059  2.469423
       ON     0.711997  0.032225  0.673416  ...     0.197050  0.658005  1.148380

[6 rows x 12 columns]

================ DRIFT SUMMARY BY STATE (ALL DEVICES) ================
          I0_nA                      ... rmse_percent                    
           mean       std       min  ...          std       min       max
state                                ...                                 
OFF    0.225772  0.017551  0.190324  ...     0.348874  1.541213  2.521148
ON     0.709039  0.018908  0.673416  ...     0.204821  0.658005  1.466079

[2 rows x 12 columns]
'''

# ============================================================
# SECTION 8D — PLOT FITTED DRIFT CURVES (one panel per device)
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True, sharey=True)

for ax, dev in zip(axes, DEVICES):
    for path in d_files:
        device, repeat = parse_drift_filename(path)
        if device != dev:
            continue

        fit = d3_fit_cache.get((device, repeat, "ON"), None)
        if fit is None:
            continue

        c = C_SERIES[(repeat - 1) % len(C_SERIES)]
        ax.plot(
            fit["t_fit"],
            fit["I_fit"] * 1e9,
            color=c,
            alpha=0.25,
            linewidth=1.0,
        )
        ax.plot(
            fit["t_fit"],
            fit["I_pred"] * 1e9,
            color=c,
            linewidth=2.0,
            label=drift_curve_label(device, repeat),
        )

    ax.set_title(dev,fontweight='bold')
    ax.grid(True)
    ax.legend(fontsize=8, loc="lower right")

axes[0].set_ylabel("|Ion| (nA)")
fig.supxlabel("Time (s)")
fig.suptitle("ON Drift Fits", fontweight='bold')
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True, sharey=True)

for ax, dev in zip(axes, DEVICES):
    for path in d_files:
        device, repeat = parse_drift_filename(path)
        if device != dev:
            continue

        fit = d3_fit_cache.get((device, repeat, "OFF"), None)
        if fit is None:
            continue

        c = C_SERIES[(repeat - 1) % len(C_SERIES)]
        ax.plot(
            fit["t_fit"],
            fit["I_fit"] * 1e9,
            color=c,
            alpha=0.25,
            linewidth=1.0,
        )
        ax.plot(
            fit["t_fit"],
            fit["I_pred"] * 1e9,
            color=c,
            linewidth=2.0,
            label=drift_curve_label(device, repeat),
        )

    ax.set_title(dev,fontweight='bold')
    ax.grid(True)
    ax.legend(fontsize=8, loc="lower right")

axes[0].set_ylabel("|Ioff| (nA)")
fig.supxlabel("Time (s)")
fig.suptitle("OFF Drift Fits", fontweight='bold')
plt.tight_layout()
plt.show()




on_summary = d3_fit_results[
    d3_fit_results["state"] == "ON"
]["a"].agg(["mean", "std", "min", "max"])

off_summary = d3_fit_results[
    d3_fit_results["state"] == "OFF"
][["A", "tau1", "B", "tau2"]].agg(["mean", "std", "min", "max"])

print("\nON power-law parameter:")
print(on_summary)

print("\nOFF double-exponential parameters:")
print(off_summary)


'''
ON power-law parameter:
mean    0.025780
std     0.002856
min     0.020074
max     0.032939
Name: a, dtype: float64

OFF double-exponential parameters:
             A       tau1         B        tau2
mean -0.429258  16.500756  0.631014  196.931349
std   0.298482   3.853963  0.270740  124.023683
min  -1.000000  12.260364  0.231730   30.905514
max  -0.206250  24.873327  1.000000  300.000000

'''

# ============================================================
# SECTION 8E — FINAL DRIFT MODEL FOR CROSSSIM
# ============================================================

OFF_FIT_TABLE_NP = np.array([
    [-0.279823, 15.791254, 0.626822, 300.000000],
    [-1.000000, 23.326950, 0.946585,  33.256634],
    [-1.000000, 24.873327, 0.962306,  36.445650],
    [-0.304159, 12.693863, 0.231730,  56.836447],
    [-0.996635, 21.183390, 0.941414,  30.905514],
    [-0.327217, 16.405843, 0.331286,  79.954135],
    [-0.271670, 13.500160, 0.704667, 300.000000],
    [-0.305352, 15.754444, 0.276956,  74.756819],
    [-0.294003, 17.096792, 0.767971, 300.000000],
    [-0.274824, 12.745730, 0.633352, 300.000000],
    [-0.206250, 13.058626, 0.383317, 300.000000],
    [-0.323915, 15.817618, 0.817728, 300.000000],
    [-0.214140, 16.471436, 0.439525, 300.000000],
    [-0.399050, 16.531542, 1.000000, 241.815029],
    [-0.241828, 12.260364, 0.401551, 300.000000],
])


def plot_crosssim_drift_model_vs_data(d_files, n_model_samples=25):
    """
    Compare final CrossSim drift model against all measured drift curves.

    ON:
        I(t)/I0 = (1 + t)^a

    OFF:
        empirical joint sampling from fitted OFF curves:
        I(t)/I0 = 1 + A*(1-exp(-t/tau1)) + B*(1-exp(-t/tau2))
    """

    t_on = np.linspace(0, FIT_T_MAX_ON, 300)
    t_off = np.linspace(0, FIT_T_MAX_OFF, 300)

    A_ON_MEAN = 0.025780
    A_ON_STD = 0.002856
    A_ON_MIN = 0.020074
    A_ON_MAX = 0.032939

    def on_crosssim_factor(t, a):
        return (1.0 + t) ** a

    def off_crosssim_factor(t, A, tau1, B, tau2):
        return (
            1.0
            + A * (1.0 - np.exp(-t / tau1))
            + B * (1.0 - np.exp(-t / tau2))
        )

    # =========================================================
    # Plot ON
    # =========================================================

    plt.figure(figsize=(8, 5))

    for path in d_files:
        df = pd.read_csv(path)

        Ion = df["ID_on"].abs()
        Ion0 = df[df["Time_on"] <= INITIAL_WINDOW_SEC]["ID_on"].abs().median()

        plt.plot(
            df["Time_on"],
            Ion / Ion0,
            color="gray",
            alpha=0.28,
            linewidth=1.0
        )

    for _ in range(n_model_samples):
        a = np.random.normal(A_ON_MEAN, A_ON_STD)
        a = np.clip(a, A_ON_MIN, A_ON_MAX)

        plt.plot(
            t_on,
            on_crosssim_factor(t_on, a),
            color="#1f4e79",
            alpha=0.22,
            linewidth=1.4
        )

    plt.plot(
        t_on,
        on_crosssim_factor(t_on, A_ON_MEAN),
        color="#0b1f3a",
        linewidth=3.0,
        label="CrossSim ON mean model"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Normalized ON current, Ion(t) / Ion0")
    plt.title("CrossSim ON Drift Model vs Measured Data")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # =========================================================
    # Plot OFF
    # =========================================================

    plt.figure(figsize=(8, 5))

    for path in d_files:
        df = pd.read_csv(path)

        Ioff = df["ID_off"].abs()
        Ioff0 = df[df["Time_off"] <= INITIAL_WINDOW_SEC]["ID_off"].abs().median()

        plt.plot(
            df["Time_off"],
            Ioff / Ioff0,
            color="gray",
            alpha=0.28,
            linewidth=1.0
        )

    for _ in range(n_model_samples):
        idx = np.random.randint(0, OFF_FIT_TABLE_NP.shape[0])

        A, tau1, B, tau2 = OFF_FIT_TABLE_NP[idx]

        plt.plot(
            t_off,
            off_crosssim_factor(t_off, A, tau1, B, tau2),
            color="#8a5a00",
            alpha=0.22,
            linewidth=1.4
        )

    # Mean of the actual fitted OFF curves, not independent mean/std sampling
    off_curves = []
    for A, tau1, B, tau2 in OFF_FIT_TABLE_NP:
        off_curves.append(off_crosssim_factor(t_off, A, tau1, B, tau2))

    off_mean_curve = np.mean(np.array(off_curves), axis=0)

    plt.plot(
        t_off,
        off_mean_curve,
        color="#3a2600",
        linewidth=3.0,
        label="CrossSim OFF empirical mean model"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Normalized OFF current, Ioff(t) / Ioff0")
    plt.title("CrossSim OFF Drift Model vs Measured Data")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_crosssim_drift_model_vs_data(d_files, n_model_samples=50)





'''


def drift_error(self, input_, time):
    """
    APM_SINW_SBFET empirical stochastic drift model.

    ON / LRS:
        I(t) = I0 * (1 + t/t0)^a

    OFF / HRS:
        Empirical joint sampling from fitted OFF curves:
        I(t) = I0 * [1 + A*(1 - exp(-t/tau1)) + B*(1 - exp(-t/tau2))]

    Time unit:
        seconds
    """

    if time == 0:
        return input_

    I = self._calculate_current(input_)

    t0 = 1.0

    # ON / LRS power-law parameters
    A_ON_MEAN = 0.025780
    A_ON_STD = 0.002856
    A_ON_MIN = 0.020074
    A_ON_MAX = 0.032939

    # OFF / HRS empirical fitted parameter table
    # Each row is one real fitted OFF drift curve: [A, tau1, B, tau2]
    OFF_FIT_TABLE = xp.array([
        [-0.279823, 15.791254, 0.626822, 300.000000],
        [-1.000000, 23.326950, 0.946585,  33.256634],
        [-1.000000, 24.873327, 0.962306,  36.445650],
        [-0.304159, 12.693863, 0.231730,  56.836447],
        [-0.996635, 21.183390, 0.941414,  30.905514],
        [-0.327217, 16.405843, 0.331286,  79.954135],
        [-0.271670, 13.500160, 0.704667, 300.000000],
        [-0.305352, 15.754444, 0.276956,  74.756819],
        [-0.294003, 17.096792, 0.767971, 300.000000],
        [-0.274824, 12.745730, 0.633352, 300.000000],
        [-0.206250, 13.058626, 0.383317, 300.000000],
        [-0.323915, 15.817618, 0.817728, 300.000000],
        [-0.214140, 16.471436, 0.439525, 300.000000],
        [-0.399050, 16.531542, 1.000000, 241.815029],
        [-0.241828, 12.260364, 0.401551, 300.000000],
    ])

    I_mid = 0.5 * (self.Imax + self.Imin)

    on_mask = I >= I_mid
    off_mask = I < I_mid

    I_drifted = xp.array(I, copy=True)

    # ON / LRS drift
    if on_mask.any():
        a_on = xp.random.normal(
            loc=A_ON_MEAN,
            scale=A_ON_STD,
            size=I[on_mask].shape
        )

        a_on = xp.clip(
            a_on,
            A_ON_MIN,
            A_ON_MAX
        )

        I_drifted[on_mask] = (
            I[on_mask]
            * (1.0 + time / t0) ** a_on
        )

    # OFF / HRS drift
    if off_mask.any():
        n_off = I[off_mask].size

        row_idx = xp.random.randint(
            0,
            OFF_FIT_TABLE.shape[0],
            size=n_off
        )

        sampled_params = OFF_FIT_TABLE[row_idx]

        A_off = sampled_params[:, 0]
        tau1_off = sampled_params[:, 1]
        B_off = sampled_params[:, 2]
        tau2_off = sampled_params[:, 3]

        off_factor = (
            1.0
            + A_off * (1.0 - xp.exp(-time / tau1_off))
            + B_off * (1.0 - xp.exp(-time / tau2_off))
        )

        I_drifted[off_mask] = I[off_mask] * off_factor

    return self._current_to_input(I_drifted)


'''







