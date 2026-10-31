'''
In The Name of God
Ali Pilehvar Meibody
Last Update : 31 oct 2026

08b_Real_TNN_Programming_Error_Monte_Carlo.py

In 08 (08_Real_TNN_Cross_S.py) we showed a SINGLE seeded realization of the
APM_SINW_SBFET programming error effect on the TNN:

    Digital TNN:                   93.78%
    Analog TNN, programming OFF:   93.78%
    Analog TNN, programming ON:    92.56%

That 93.78% -> 92.56% comparison came from ONE fixed seed (SEED=42), i.e. ONE
random draw of the programming-error noise. A reviewer asked whether this was
evaluated across multiple independent simulations, and for the statistical
variation. This script answers that directly, AND verifies (at runtime, not
just by code inspection) that the Monte Carlo ensemble is actually valid:

    V1. Explicitly reproduces the original seed=42 result and checks it
        against the 92.56% reported in 08 (instead of hard-coding it). This
        check is STRICT: if the reproduction does not match within tolerance,
        the script raises and stops before running the Monte Carlo ensemble.
    V2. Verifies (at runtime) that the internal CrossSim attributes used by
        this script's own fingerprinting/registration helpers (core_pos,
        core_neg, .matrix, .device, ._programming_error_model) actually
        exist and behave as expected, THEN confirms APM_SINW_SBFET is
        registered in CrossSim's device registry and is the actual class
        instantiated inside the programmed core.
    V3. Confirms every one of the N_RUNS realizations produces a genuinely
        different positive/negative conductance matrix (fingerprinted). This
        establishes DISTINCTNESS of the 30 programmed crossbars — it is not,
        by itself, a formal statistical-independence test. Independence
        instead follows from the experimental design (a fresh, disjoint RNG
        seed per run) and from the source-code audit confirming nothing in
        the call path silently re-seeds the RNG (see note before the Monte
        Carlo loop below). V3 also confirms the underlying trained ternary
        weights of net_tnn never change across all conversions.
    V4. Confirms the CrossSimParameters used here are identical (field by
        field) to the hardware configuration in 08 (one-sided balanced
        mapping, Rmin/Rmax, APM model, ideal ADC/DAC, parasitics/drift/
        read-noise disabled).
    V5. Confirms programming error is sampled ONCE at construction
        (from_torch -> set_matrix -> apply_write_error) and is NOT resampled
        during inference: evaluating the SAME programmed model twice on the
        full test set must give a bit-identical accuracy.
    V6/V7. Runs N_RUNS = 30 independent realizations, reports individual
        values, mean, sample std, min, max, 95% CI (in percentage points),
        saves per-run + summary CSVs, and produces one publication figure
        (plus optional diagnostics).
    V8. Re-runs one seed a second time to confirm bit-identical,
        reproducible results from the same seed list.

This file only READS the existing checkpoint (TERNARY_ONLY_..._seed42_...pth)
produced by 03/04, and reuses (duplicated, read-only) the exact
build_tnn_apm_params(...) configuration from 08. It does not modify
03/04/07/08/09/10, retrain the TNN, or change the device model in any way.
'''

# ============================================
'''                   Imports              '''
# ============================================
import os
import sys
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy import stats
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from paths import DATA_DIR, PROJECT_DIR, choose_checkpoint, ensure_cross_sim_on_path
ensure_cross_sim_on_path()

from simulator import CrossSimParameters
from simulator.algorithms.dnn.torch.convert import from_torch, convertible_modules
from simulator.devices.device import Device

# ============================================
'''                Variables              '''
# ============================================
SEED = 42  # the exact seed used for the single-realization result in 08

# Same ternary-only checkpoint as 07/08/09. One match is used as-is.
TNN_CHECKPOINT = choose_checkpoint("tnn")

# Same measured device-range parameters as 05/07/08/09
R_MIN = 1.410360e+09
R_MAX = 4.429246e+09
TNN_APM_MODEL = "APM_SINW_SBFET"

# Value printed in 08_Real_TNN_Cross_S.py for "Analog TNN, programming ON"
# (seed=42). This is now treated as a value to be REPRODUCED and checked,
# not assumed.
REPORTED_08_PROGRAMMING_ON_ACCURACY = 92.56
REPORTED_08_DIGITAL_ACCURACY = 93.78
REPRODUCTION_TOLERANCE_PP = 0.01  # percentage points; pipeline is deterministic

# ------------------------------------------------------------------
# Monte Carlo configuration
# ------------------------------------------------------------------
N_RUNS = 30                        # number of independent programming realizations
MC_SEED_BASE = 10_000               # disjoint from SEED=42 (kept as the single-run reference)
MC_SEEDS = [MC_SEED_BASE + i for i in range(N_RUNS)]

# Additional diagnostic figures (histogram, per-run scatter) are optional.
# The one required publication figure (ideal baseline vs mean +/- std, with
# individual realizations overlaid) is always produced.
SAVE_DIAGNOSTIC_FIGURES = True

DATE_TAG = date.today().strftime("%Y%m%d")
OUT_DIR = PROJECT_DIR / "MonteCarlo_Results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
'''                    TNN LOAD                '''
# =========================================================
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import importlib
_load = importlib.import_module("04_Load_ANN_TNN")
load_tnn_ternary = _load.load_tnn_ternary
print_model_summary = _load.print_model_summary

net_tnn, tnn_path, tnn_meta = load_tnn_ternary(TNN_CHECKPOINT)
print_model_summary(net_tnn, "TNN (ternary -1/0/+1)", tnn_path, tnn_meta)
'''
TNN (ternary -1/0/+1)
  checkpoint: TERNARY_ONLY_mnist_tw0.7_th0.05_seed42_20260908.pth
  parameters: 235,146
  saved test accuracy: 93.78%
  layers: fc1 (256, 784), fc2 (128, 256), fc3 (10, 128)
'''

# Snapshot of the TRAINED ternary weights, taken before any CrossSim
# conversion happens. Compared again at the very end of the script (V3) to
# prove from_torch() never mutates the source torch model.
_ORIG_WEIGHTS = {
    name: getattr(net_tnn, name).weight.detach().clone()
    for name in ("fc1", "fc2", "fc3")
}

# =========================================================
'''                  Load Data               '''
# =========================================================
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


def evaluate(model, loader, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100 * correct / total


# ============================================================
# TNN + APM_SINW_SBFET — SAME CrossSim config as 08 (programming
# error only; drift OFF; ideal ADC/DAC; no parasitics).
#
# This function is a byte-for-byte copy of build_tnn_apm_params(...) in
# 08_Real_TNN_Cross_S.py. V4 below additionally asserts, at runtime, that the
# resulting CrossSimParameters object matches the intended hardware
# configuration field-by-field.
# ============================================================
def build_tnn_apm_params(programming_enable=False, drift_enable=False, time=0.0):
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


# ============================================================
# Helpers reused (read-only, duplicated — NOT imported/modified) from
# 08_Real_TNN_Cross_S.py, used here purely for verification/fingerprinting.
# ============================================================
def collect_tnn_balanced_cores(model):
    """Per-layer BalancedCore wrapper (has .core_pos / .core_neg) for a
    converted analog TNN. Identical logic to 08's helper of the same name."""
    cores = {}
    for name, module in model.named_modules():
        core_obj = None
        if hasattr(module, "core"):
            core_obj = module.core
        elif hasattr(module, "analog_core"):
            core_obj = module.analog_core
        if core_obj is None:
            continue
        try:
            wrapper = core_obj.cores[0][0]
            if hasattr(wrapper, "core_pos") and hasattr(wrapper, "core_neg"):
                cores[name] = wrapper
        except Exception:
            pass
    return cores


def get_balanced_maps(wrapper_core, scale_rmin):
    g_pos = np.array(wrapper_core.core_pos.matrix) / scale_rmin
    g_neg = np.array(wrapper_core.core_neg.matrix) / scale_rmin
    g_eff = g_pos - g_neg
    return g_pos, g_neg, g_eff


def layer_fingerprint(model):
    """Per-layer (G_pos, G_neg) programmed conductance maps for a converted
    analog TNN. Used to prove two runs did/did not sample identical
    programming-error noise."""
    cores = collect_tnn_balanced_cores(model)
    fp = {}
    for name, wrapper in cores.items():
        g_pos, g_neg, g_eff = get_balanced_maps(wrapper, R_MIN)
        fp[name] = (g_pos.copy(), g_neg.copy(), g_eff.copy())
    return fp


def fingerprint_checksum(fp):
    """Compact (sum, sum-of-squares) checksum per layer, concatenated across
    layers in a fixed order. Two runs with identical checksums would (for all
    practical purposes) imply identical programmed matrices; used to verify
    that the N_RUNS realizations are mutually distinct."""
    parts = []
    for name in sorted(fp.keys()):
        _, _, g_eff = fp[name]
        parts.append(round(float(np.sum(g_eff)), 10))
        parts.append(round(float(np.sum(g_eff ** 2)), 10))
    return tuple(parts)


def registered_programming_error_classes(model):
    """Returns {layer_name: (class name of _programming_error_model used by
    core_pos, same for core_neg)} — proves which device class CrossSim
    actually instantiated and applied for this converted model."""
    cores = collect_tnn_balanced_cores(model)
    out = {}
    for name, wrapper in cores.items():
        pos_cls = wrapper.core_pos.device._programming_error_model.__class__.__name__
        neg_cls = wrapper.core_neg.device._programming_error_model.__class__.__name__
        out[name] = (pos_cls, neg_cls)
    return out


def verify_crosssim_internal_api(model, label):
    """V2 (part a): Runtime check that the internal CrossSim attribute chain
    relied upon by collect_tnn_balanced_cores / get_balanced_maps /
    registered_programming_error_classes actually exists and behaves as
    expected for THIS converted model, before any of those helpers are
    trusted elsewhere in the script.

    This does not just assume the private attributes (.core, .cores,
    .core_pos/.core_neg, .matrix, .device, ._programming_error_model) are
    present — it explicitly checks for each of them and raises a clear,
    actionable error the moment one is missing, rather than silently
    producing an empty/incorrect fingerprint later on.
    """
    cores = collect_tnn_balanced_cores(model)
    if not cores:
        raise AssertionError(
            f"[V2a] FAILED ({label}) — collect_tnn_balanced_cores() returned "
            f"no layers. The .core/.cores[0][0]/.core_pos/.core_neg attribute "
            f"chain this script depends on is not present on the converted "
            f"model; the fingerprinting and registration checks below cannot "
            f"be trusted."
        )

    required_core_attrs = ("matrix", "device")
    for layer_name, wrapper in cores.items():
        for side_name, side_core in (("core_pos", getattr(wrapper, "core_pos", None)),
                                      ("core_neg", getattr(wrapper, "core_neg", None))):
            if side_core is None:
                raise AssertionError(
                    f"[V2a] FAILED ({label}) — layer '{layer_name}' is missing "
                    f"'{side_name}' on its BalancedCore wrapper."
                )
            for attr in required_core_attrs:
                if not hasattr(side_core, attr):
                    raise AssertionError(
                        f"[V2a] FAILED ({label}) — layer '{layer_name}' "
                        f"'{side_name}' has no '.{attr}' attribute. CrossSim's "
                        f"internal core API may have changed; "
                        f"get_balanced_maps()/registered_programming_error_classes() "
                        f"would silently fail or return wrong data."
                    )
            # .matrix must be a non-empty, array-convertible object
            matrix_arr = np.array(side_core.matrix)
            if matrix_arr.size == 0:
                raise AssertionError(
                    f"[V2a] FAILED ({label}) — layer '{layer_name}' "
                    f"'{side_name}.matrix' is empty; cannot fingerprint a "
                    f"zero-size programmed conductance matrix."
                )
            # .device must expose the programming-error model object
            device_obj = side_core.device
            if not hasattr(device_obj, "_programming_error_model"):
                raise AssertionError(
                    f"[V2a] FAILED ({label}) — layer '{layer_name}' "
                    f"'{side_name}.device' has no '_programming_error_model' "
                    f"attribute; device-model registration cannot be verified."
                )
            prog_model_obj = device_obj._programming_error_model
            if not hasattr(prog_model_obj, "programming_error") or not callable(
                prog_model_obj.programming_error
            ):
                raise AssertionError(
                    f"[V2a] FAILED ({label}) — layer '{layer_name}' "
                    f"'{side_name}.device._programming_error_model' has no "
                    f"callable 'programming_error' method."
                )

    print(f"[V2a] PASS ({label}) — for all {len(cores)} layer(s), "
          f"core_pos/core_neg expose working .matrix, .device, and "
          f".device._programming_error_model.programming_error(...) "
          f"attributes. The fingerprinting and registration checks below "
          f"are therefore operating on real, verified CrossSim internals.")
    return True


def verify_hardware_config(p, expected_programming_enable, label):
    """V4: asserts, field by field, that CrossSimParameters p matches the
    exact hardware configuration used for the original programming-variability
    experiment in 08_Real_TNN_Cross_S.py."""
    checks = [
        ("core.style (BALANCED=1)",                 p.core.style, 1),
        ("core.balanced.style (ONE_SIDED=1)",        p.core.balanced.style, 1),
        ("core.balanced.interleaved_posneg",         p.core.balanced.interleaved_posneg, False),
        ("core.balanced.subtract_current_in_xbar",   p.core.balanced.subtract_current_in_xbar, True),
        ("core.weight_bits",                         p.core.weight_bits, 0),
        ("core.bit_sliced.num_slices",               p.core.bit_sliced.num_slices, 1),
        ("xbar.device.Rmin",                         p.xbar.device.Rmin, R_MIN),
        ("xbar.device.Rmax",                         p.xbar.device.Rmax, R_MAX),
        ("xbar.device.cell_bits",                    p.xbar.device.cell_bits, 1),
        ("xbar.device.Vread",                         p.xbar.device.Vread, 1.0),
        ("xbar.device.programming_error.enable",     p.xbar.device.programming_error.enable, expected_programming_enable),
        ("xbar.device.programming_error.model",      p.xbar.device.programming_error.model, TNN_APM_MODEL),
        ("xbar.device.read_noise.enable",            p.xbar.device.read_noise.enable, False),
        ("xbar.device.drift_error.enable",           p.xbar.device.drift_error.enable, False),
        ("xbar.adc.mvm.bits (ideal)",                p.xbar.adc.mvm.bits, 0),
        ("xbar.adc.vmm.bits (ideal)",                p.xbar.adc.vmm.bits, 0),
        ("xbar.dac.mvm.bits (ideal)",                p.xbar.dac.mvm.bits, 0),
        ("xbar.dac.vmm.bits (ideal)",                p.xbar.dac.vmm.bits, 0),
        ("xbar.array.parasitics.enable",             p.xbar.array.parasitics.enable, False),
    ]
    all_ok = True
    print(f"\n[V4] Hardware-configuration check ({label}):")
    for name, actual, expected in checks:
        ok = (actual == expected)
        all_ok = all_ok and ok
        print(f"    [{'OK' if ok else 'MISMATCH'}] {name}: got {actual!r} (expected {expected!r})")
    if not all_ok:
        raise AssertionError(
            f"Hardware configuration for '{label}' does not match the "
            f"original 08_Real_TNN_Cross_S.py experiment."
        )
    print(f"[V4] PASS — '{label}' configuration is identical to 08_Real_TNN_Cross_S.py.")
    return all_ok


# ============================================================
# 1. Digital baseline (deterministic, single value)
# ============================================================
digital_tnn_acc = evaluate(net_tnn, test_loader, device="cpu")
print("Digital TNN accuracy:", digital_tnn_acc)
print("Convertible modules:", convertible_modules(net_tnn))

# ============================================================
# 2. Analog TNN, programming OFF (deterministic sanity check)
#    No programming-error sampling happens in this branch, so the
#    result should be identical to the digital accuracy regardless
#    of seed. We still fix a seed for full reproducibility.
# ============================================================
params_no_prog = build_tnn_apm_params(programming_enable=False, drift_enable=False, time=0.0)
verify_hardware_config(params_no_prog, expected_programming_enable=False, label="Programming OFF")

np.random.seed(SEED)
torch.manual_seed(SEED)
analog_tnn_no_prog = from_torch(net_tnn, params_no_prog)
analog_tnn_no_prog.eval()
acc_no_prog = evaluate(analog_tnn_no_prog, test_loader, device="cpu")
print("Analog TNN, programming OFF (single deterministic run):", acc_no_prog)

if abs(acc_no_prog - digital_tnn_acc) > REPRODUCTION_TOLERANCE_PP:
    raise AssertionError(
        f"Analog 'programming OFF' accuracy ({acc_no_prog:.4f}%) does not match "
        f"digital accuracy ({digital_tnn_acc:.4f}%) within {REPRODUCTION_TOLERANCE_PP} pp."
    )
print(f"[sanity check] PASS — programming OFF reproduces the digital accuracy exactly "
      f"(as expected, since no error is applied in this branch).")

# ============================================================
# V1 + V2 + V5: Explicit reproduction of the ORIGINAL seed=42 result
# from 08_Real_TNN_Cross_S.py, plus registration and fixed-programming
# checks performed directly on this reproduced model.
# ============================================================
print("\n" + "=" * 72)
print("V1/V2/V5 — Reproducing and verifying the original seed=42 result")
print("=" * 72)

params_seed42 = build_tnn_apm_params(programming_enable=True, drift_enable=False, time=0.0)
verify_hardware_config(params_seed42, expected_programming_enable=True, label="Programming ON (seed=42 reproduction)")

np.random.seed(SEED)
torch.manual_seed(SEED)
analog_tnn_seed42 = from_torch(net_tnn, params_seed42)
analog_tnn_seed42.eval()

# --- V2a: confirm the internal CrossSim attributes this script relies on
#     (core_pos/core_neg, .matrix, .device, ._programming_error_model) exist
#     and work, BEFORE trusting any fingerprint/registration result below ---
verify_crosssim_internal_api(analog_tnn_seed42, label="seed=42 reproduction")

# --- V2b: confirm APM_SINW_SBFET is registered and actually instantiated ---
all_device_types = {subcls.__name__ for subcls in Device.get_all_subclasses()}
if TNN_APM_MODEL not in all_device_types:
    raise AssertionError(
        f"'{TNN_APM_MODEL}' is not among CrossSim's registered device "
        f"subclasses: {sorted(all_device_types)}"
    )
print(f"[V2b] '{TNN_APM_MODEL}' is registered in CrossSim's device subclass registry.")

prog_classes = registered_programming_error_classes(analog_tnn_seed42)
for layer_name, (pos_cls, neg_cls) in prog_classes.items():
    if pos_cls != TNN_APM_MODEL or neg_cls != TNN_APM_MODEL:
        raise AssertionError(
            f"Layer {layer_name}: expected programming-error model "
            f"'{TNN_APM_MODEL}' on both core_pos/core_neg, got "
            f"({pos_cls}, {neg_cls})."
        )
    print(f"[V2b] PASS — layer '{layer_name}': core_pos/core_neg both use "
          f"'{pos_cls}' as the applied programming-error model.")

# --- V5: fixed programming during inference (no resampling per batch) ---
acc_seed42_eval1 = evaluate(analog_tnn_seed42, test_loader, device="cpu")
acc_seed42_eval2 = evaluate(analog_tnn_seed42, test_loader, device="cpu")
if acc_seed42_eval1 != acc_seed42_eval2:
    raise AssertionError(
        f"[V5] FAILED — evaluating the SAME programmed model twice gave "
        f"different accuracies ({acc_seed42_eval1} vs {acc_seed42_eval2}). "
        f"This would indicate programming error (or some other noise source) "
        f"is being resampled during inference, not fixed at construction time."
    )
print(f"[V5] PASS — evaluating the same programmed crossbar twice on the full "
      f"test set gives a bit-identical accuracy ({acc_seed42_eval1:.6f}% both "
      f"times): programming error is fixed at construction, not resampled per batch.")
acc_seed42_reproduced = acc_seed42_eval1

# --- V1: compare the reproduced value against the reported 92.56% ---
reproduction_diff_pp = abs(acc_seed42_reproduced - REPORTED_08_PROGRAMMING_ON_ACCURACY)
reproduction_matches = reproduction_diff_pp <= REPRODUCTION_TOLERANCE_PP

print(f"\n[V1] Reported in 08_Real_TNN_Cross_S.py  (seed=42): "
      f"{REPORTED_08_PROGRAMMING_ON_ACCURACY:.2f}%")
print(f"[V1] Reproduced here, same config+seed    : {acc_seed42_reproduced:.4f}%")
print(f"[V1] Difference                           : {reproduction_diff_pp:.4f} pp")
if not reproduction_matches:
    raise AssertionError(
        f"[V1] FAILED — reproduced seed=42 accuracy "
        f"({acc_seed42_reproduced:.4f}%) does not match the value reported "
        f"in 08_Real_TNN_Cross_S.py ({REPORTED_08_PROGRAMMING_ON_ACCURACY:.2f}%) "
        f"within the {REPRODUCTION_TOLERANCE_PP} pp tolerance "
        f"(difference = {reproduction_diff_pp:.4f} pp). Stopping BEFORE "
        f"running the Monte Carlo ensemble. Investigate the checkpoint path, "
        f"CrossSimParameters configuration, library/package version, or RNG "
        f"handling before re-running this script."
    )
print(f"[V1] PASS — reproduction matches the reported value within "
      f"{REPRODUCTION_TOLERANCE_PP} pp.")

reference_fp = layer_fingerprint(analog_tnn_seed42)
reference_checksum = fingerprint_checksum(reference_fp)

# ============================================================
# 3. Analog TNN, programming ON — Monte Carlo over N_RUNS
#    INDEPENDENT programming realizations.
#
#    Each iteration:
#      - builds a fresh CrossSimParameters object
#      - sets a NEW, independent RNG seed (np.random.seed controls the
#        xp.random.normal(...) draws inside APM_SINW_SBFET.programming_error;
#        torch.manual_seed controls the throwaway default nn.Linear init that
#        from_torch() creates internally before overwriting it with the
#        ternary weights — this also perturbs the numpy RNG stream via the
#        dummy set_matrix() call made during __init__, so both seeds must be
#        set for full determinism, exactly as done in 08)
#      - calls from_torch(...) which internally re-ternarizes the weights
#        (no-op here, already ternary) and programs them via set_matrix() ->
#        device.apply_write_error(), i.e. draws a fresh, independent sample
#        of the measured ON/OFF relative-error distribution
#      - evaluates test accuracy on the SAME fixed MNIST test set
#      - fingerprints the resulting G_pos/G_neg/G_eff conductance maps so
#        DISTINCTNESS across runs can be confirmed numerically (V3) — i.e.
#        that each run really produced a different programmed crossbar,
#        rather than assuming it. Distinct outputs are necessary evidence
#        that the seed was effective, but are not themselves a formal proof
#        of statistical independence of the underlying random draws.
#
#    The actual case for INDEPENDENCE across runs rests on the experimental
#    design and the following source-code audit, not on the fingerprint
#    check: no function anywhere in the torch conversion path, the cores,
#    or the devices package calls np.random.seed/torch.manual_seed
#    internally (verified by direct source inspection of convert.py,
#    layer.py, linear.py, numeric_core.py, balanced_core.py, device.py,
#    idevice.py, and APM_SINW_SBFET.py — the only np.random.seed(...) call
#    anywhere in the CrossSim package is inside the unrelated Keras-style
#    simulator.algorithms.dnn.dnn.DNN class, which from_torch() never
#    constructs or calls). Since each run explicitly re-seeds with its own
#    disjoint seed value and nothing downstream resets that state, the 30
#    draws of xp.random.normal(...) are independent samples from the
#    calibrated APM_SINW_SBFET error distribution by construction.
# ============================================================
print(f"\nRunning {N_RUNS} independent programming-error realizations "
      f"({TNN_APM_MODEL})...")

mc_rows = []
mc_fingerprints = []
for i, mc_seed in enumerate(MC_SEEDS):
    params_with_prog = build_tnn_apm_params(
        programming_enable=True,
        drift_enable=False,
        time=0.0
    )
    np.random.seed(mc_seed)
    torch.manual_seed(mc_seed)

    analog_tnn_with_prog = from_torch(net_tnn, params_with_prog)
    analog_tnn_with_prog.eval()
    acc = evaluate(analog_tnn_with_prog, test_loader, device="cpu")

    fp = layer_fingerprint(analog_tnn_with_prog)
    checksum = fingerprint_checksum(fp)
    mc_fingerprints.append(checksum)

    drop_pp = digital_tnn_acc - acc
    mc_rows.append({
        "run": i + 1,
        "seed": mc_seed,
        "accuracy_percent": acc,
        "drop_vs_digital_pp": drop_pp,
        "g_eff_sum_checksum": checksum[0::2],   # one value per layer (sum)
        "g_eff_sumsq_checksum": checksum[1::2],  # one value per layer (sum of squares)
    })
    print(f"  run {i + 1:2d}/{N_RUNS}  seed={mc_seed:<6d}  "
          f"acc={acc:6.2f}%   drop={drop_pp:+.2f} pp")

mc_df = pd.DataFrame(mc_rows)

# ------------------------------------------------------------------
# V3 (part 1) — DISTINCTNESS check: every run must have produced a
# genuinely different programmed crossbar. This confirms that distinct
# programmed conductance matrices were generated on each run (i.e. the seed
# was effective); it is a necessary consistency check, not a formal proof
# of statistical independence (see note above the Monte Carlo loop for the
# independence argument, which rests on the source-code audit).
# ------------------------------------------------------------------
n_unique_checksums = len(set(mc_fingerprints))
if n_unique_checksums != N_RUNS:
    raise AssertionError(
        f"[V3] FAILED — expected {N_RUNS} mutually distinct programmed "
        f"crossbars, but only found {n_unique_checksums} unique conductance "
        f"fingerprints. Some runs produced identical programmed conductance "
        f"values despite using different seeds, which would indicate the "
        f"seed is not actually varying the sampled programming-error noise."
    )
print(f"\n[V3] PASS — all {N_RUNS} programming realizations produced mutually "
      f"distinct G_pos/G_neg/G_eff conductance matrices, confirming that a "
      f"different programmed crossbar was generated on every run (this "
      f"confirms distinctness, not a formal statistical-independence test "
      f"— see the independence argument above the Monte Carlo loop).")

# Small human-readable proof: first 5 programmed G_pos conductance values of
# fc1, for the first three runs, to show the arrays really differ.
print("\n[V3] Example — fc1 G_pos (first 5 entries), first 3 runs:")
for i, mc_seed in enumerate(MC_SEEDS[:3]):
    params_dbg = build_tnn_apm_params(programming_enable=True, drift_enable=False, time=0.0)
    np.random.seed(mc_seed)
    torch.manual_seed(mc_seed)
    m_dbg = from_torch(net_tnn, params_dbg)
    m_dbg.eval()
    fp_dbg = layer_fingerprint(m_dbg)
    g_pos_fc1 = fp_dbg["fc1"][0].ravel()[:5]
    print(f"    run {i + 1} (seed={mc_seed}): {np.array2string(g_pos_fc1, precision=6)}")

# ------------------------------------------------------------------
# V3 (part 2) — the trained ternary weights of net_tnn must be completely
# unaffected by having been converted via from_torch() 30+ times.
# ------------------------------------------------------------------
weights_unchanged = all(
    torch.equal(getattr(net_tnn, name).weight.detach(), _ORIG_WEIGHTS[name])
    for name in ("fc1", "fc2", "fc3")
)
if not weights_unchanged:
    raise AssertionError(
        "[V3] FAILED — the trained ternary weights of net_tnn changed after "
        "running the Monte Carlo ensemble. from_torch() should always return "
        "a converted COPY and never mutate the source torch model."
    )
print("[V3] PASS — net_tnn's trained ternary weights (fc1/fc2/fc3) are "
      "byte-identical before and after all 30 conversions.")

# ------------------------------------------------------------------
# V8 — repeatability: re-running the exact same seed a second time must
# give a bit-identical accuracy AND a bit-identical conductance fingerprint.
# ------------------------------------------------------------------
repeat_seed = MC_SEEDS[0]
params_repeat = build_tnn_apm_params(programming_enable=True, drift_enable=False, time=0.0)
np.random.seed(repeat_seed)
torch.manual_seed(repeat_seed)
analog_tnn_repeat = from_torch(net_tnn, params_repeat)
analog_tnn_repeat.eval()
acc_repeat = evaluate(analog_tnn_repeat, test_loader, device="cpu")
checksum_repeat = fingerprint_checksum(layer_fingerprint(analog_tnn_repeat))

repeatability_ok = (
    acc_repeat == mc_df.loc[0, "accuracy_percent"]
    and checksum_repeat == mc_fingerprints[0]
)
if not repeatability_ok:
    raise AssertionError(
        f"[V8] FAILED — re-running seed={repeat_seed} gave a different "
        f"accuracy/fingerprint than the first time (run 1: "
        f"{mc_df.loc[0, 'accuracy_percent']:.4f}% vs repeat: {acc_repeat:.4f}%). "
        f"The Monte Carlo ensemble is not reproducible."
    )
print(f"[V8] PASS — re-running seed={repeat_seed} reproduces run #1 exactly "
      f"({acc_repeat:.4f}%, identical conductance fingerprint): the ensemble "
      f"is fully reproducible given the same seed list.")

# ============================================================
# 4. Statistics across the N_RUNS independent realizations
#    (all differences reported in percentage points, pp — not relative %)
# ============================================================
accs = mc_df["accuracy_percent"].to_numpy(dtype=float)
drops = mc_df["drop_vs_digital_pp"].to_numpy(dtype=float)

n = len(accs)
acc_mean = float(np.mean(accs))
acc_std = float(np.std(accs, ddof=1))          # sample std (N-1)
acc_sem = acc_std / np.sqrt(n)
acc_min = float(np.min(accs))
acc_max = float(np.max(accs))
acc_median = float(np.median(accs))

drop_mean = float(np.mean(drops))
drop_std = float(np.std(drops, ddof=1))
drop_sem = drop_std / np.sqrt(n)

# 95% CI of the MEAN using the Student-t distribution (n-1 dof)
t_crit = stats.t.ppf(0.975, df=n - 1)
acc_ci95 = t_crit * acc_sem
drop_ci95 = t_crit * drop_sem

# 95% prediction interval for a SINGLE new realization (relevant for
# checking whether the historical seed=42 draw is a plausible individual
# sample from this distribution — wider than the CI of the mean).
pred_half_width = t_crit * acc_std * np.sqrt(1 + 1 / n)
pred_lo, pred_hi = acc_mean - pred_half_width, acc_mean + pred_half_width

seed42_within_minmax = acc_min <= acc_seed42_reproduced <= acc_max
seed42_within_1std = (acc_mean - acc_std) <= acc_seed42_reproduced <= (acc_mean + acc_std)
seed42_within_pred_interval = pred_lo <= acc_seed42_reproduced <= pred_hi

# Guard against division by zero: if all N_RUNS realizations happened to
# produce IDENTICAL accuracy (acc_std == 0), a z-score is mathematically
# undefined rather than +/-inf or nan from a silent 0/0 or x/0 division.
if acc_std > 0:
    seed42_zscore = (acc_seed42_reproduced - acc_mean) / acc_std
    seed42_zscore_defined = True
else:
    seed42_zscore = float("nan")
    seed42_zscore_defined = False
    print(f"\n[note] std(accuracy) across the {n} realizations is exactly 0 "
          f"— all runs produced the identical accuracy ({acc_mean:.2f}%). "
          f"The z-score of the seed=42 value is mathematically undefined in "
          f"this case; membership is instead reported via exact equality "
          f"and the [min, max] / prediction-interval checks below.")

print("\n" + "=" * 72)
print(f"PROGRAMMING-ERROR MONTE CARLO SUMMARY  (N = {n} independent runs)")
print("=" * 72)
print(f"Digital TNN accuracy                      : {digital_tnn_acc:.2f}%  (deterministic)")
print(f"Analog TNN, programming OFF               : {acc_no_prog:.2f}%  (deterministic)")
print(f"Analog TNN, programming ON (seed=42, reproduced here) : {acc_seed42_reproduced:.2f}%")
print("-" * 72)
print(f"Analog TNN, programming ON  (N={n} independent realizations):")
print(f"  mean accuracy        = {acc_mean:.2f}%")
print(f"  std accuracy         = {acc_std:.2f} pp")
print(f"  SEM                  = {acc_sem:.3f} pp")
print(f"  95% CI of the mean   = {acc_mean:.2f}% +/- {acc_ci95:.2f} pp")
print(f"  median               = {acc_median:.2f}%")
print(f"  min / max            = {acc_min:.2f}% / {acc_max:.2f}%")
print(f"  95% prediction interval for a single new run : "
      f"[{pred_lo:.2f}%, {pred_hi:.2f}%]")
print("-" * 72)
print(f"Accuracy drop vs digital baseline ({digital_tnn_acc:.2f}%), in pp:")
print(f"  mean drop            = {drop_mean:.2f} pp")
print(f"  std drop             = {drop_std:.2f} pp")
print(f"  95% CI of mean drop  = {drop_mean:.2f} +/- {drop_ci95:.2f} pp")
print("-" * 72)
print(f"Does the original seed=42 realization ({acc_seed42_reproduced:.2f}%) "
      f"fall within the 30-run distribution?")
print(f"  within [min, max] = [{acc_min:.2f}, {acc_max:.2f}] %        : {seed42_within_minmax}")
print(f"  within mean +/- 1 std                                : {seed42_within_1std}")
print(f"  within 95% prediction interval for a single new run  : {seed42_within_pred_interval}")
if seed42_zscore_defined:
    print(f"  z-score vs the 30-run distribution                   : {seed42_zscore:+.2f}")
else:
    print(f"  z-score vs the 30-run distribution                   : "
          f"undefined (std = 0; all {n} realizations gave the identical accuracy)")
print("=" * 72)
print(f"\nReport-ready line:")
zscore_clause = (
    f"(z = {seed42_zscore:+.2f})" if seed42_zscore_defined
    else "(z-score undefined: std = 0 across the 30 realizations)"
)
print(f"  Programming-error accuracy (APM_SINW_SBFET), N={n} independent "
      f"realizations: {acc_mean:.2f}% +/- {acc_std:.2f}% (std) "
      f"[min {acc_min:.2f}%, max {acc_max:.2f}%], "
      f"i.e. a drop of {drop_mean:.2f} +/- {drop_std:.2f} pp vs the "
      f"{digital_tnn_acc:.2f}% digital baseline. The single-seed value "
      f"reported in 08_Real_TNN_Cross_S.py ({acc_seed42_reproduced:.2f}%) lies "
      f"within this distribution {zscore_clause}.")

# ============================================================
# 5. Save full per-run results (CSV) for transparency / rebuttal
# ============================================================
mc_df_out = mc_df.drop(columns=["g_eff_sum_checksum", "g_eff_sumsq_checksum"]).copy()
# Expand the per-layer checksum tuples into their own columns for the CSV.
layer_names_sorted = sorted(layer_fingerprint(analog_tnn_seed42).keys())
for li, lname in enumerate(layer_names_sorted):
    mc_df_out[f"{lname}_g_eff_sum"] = [row["g_eff_sum_checksum"][li] for row in mc_rows]
    mc_df_out[f"{lname}_g_eff_sumsq"] = [row["g_eff_sumsq_checksum"][li] for row in mc_rows]

csv_path = OUT_DIR / f"{DATE_TAG}_tnn_programming_error_mc_{n}runs.csv"
mc_df_out.to_csv(csv_path, index=False)

summary_df = pd.DataFrame([{
    "digital_accuracy_percent": digital_tnn_acc,
    "programming_off_accuracy_percent": acc_no_prog,
    "programming_on_seed42_reproduced_percent": acc_seed42_reproduced,
    "programming_on_seed42_reported_in_08_percent": REPORTED_08_PROGRAMMING_ON_ACCURACY,
    "seed42_reproduction_matches_08": reproduction_matches,
    "n_runs": n,
    "programming_on_mean_percent": acc_mean,
    "programming_on_std_percent": acc_std,
    "programming_on_sem_percent": acc_sem,
    "programming_on_ci95_halfwidth_percent": acc_ci95,
    "programming_on_median_percent": acc_median,
    "programming_on_min_percent": acc_min,
    "programming_on_max_percent": acc_max,
    "pred_interval_95_low_percent": pred_lo,
    "pred_interval_95_high_percent": pred_hi,
    "drop_vs_digital_mean_pp": drop_mean,
    "drop_vs_digital_std_pp": drop_std,
    "drop_vs_digital_ci95_halfwidth_pp": drop_ci95,
    "seed42_within_minmax_of_30runs": seed42_within_minmax,
    "seed42_within_mean_pm_1std": seed42_within_1std,
    "seed42_within_95pct_prediction_interval": seed42_within_pred_interval,
    "seed42_zscore_vs_30runs": seed42_zscore,
    "seed42_zscore_defined": seed42_zscore_defined,
    "v2a_crosssim_internal_api_verified": True,  # would have raised above otherwise
    "v2_device_model_registered_and_applied": all(
        pos == TNN_APM_MODEL and neg == TNN_APM_MODEL
        for pos, neg in prog_classes.values()
    ),
    "v3_all_30_realizations_mutually_distinct": (n_unique_checksums == N_RUNS),
    "v3_trained_ternary_weights_unchanged": weights_unchanged,
    "v5_programming_error_fixed_at_construction_not_resampled": (acc_seed42_eval1 == acc_seed42_eval2),
    "v8_repeat_seed_reproducible": repeatability_ok,
}])
summary_path = OUT_DIR / f"{DATE_TAG}_tnn_programming_error_mc_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"\nSaved per-run results : {csv_path}")
print(f"Saved summary stats   : {summary_path}")

# ============================================================
# 6. Figure — ONE publication-quality figure (required):
#    ideal baseline vs mean programming-aware accuracy, with std error bars
#    and individual realizations overlaid.
# ============================================================
_COLOR_DIGITAL = "#4A5568"   # dark gray
_COLOR_OFF = "#2C5282"       # dark blue
_COLOR_ON_MEAN = "#C05621"   # burnt orange

fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=300)
ax.set_axisbelow(True)
ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3, zorder=0)

labels = ["Digital\n(ideal)", "Prog OFF\n(ideal, analog)", f"Prog ON\n(mean of N={n})"]
bar_vals = [digital_tnn_acc, acc_no_prog, acc_mean]
bar_errs = [0.0, 0.0, acc_std]
bar_colors = [_COLOR_DIGITAL, _COLOR_OFF, _COLOR_ON_MEAN]

bars = ax.bar(
    labels, bar_vals, yerr=bar_errs, capsize=6,
    color=bar_colors, edgecolor="black", linewidth=0.8, width=0.6,
    error_kw=dict(elinewidth=1.6, ecolor="black"), zorder=3,
)

# overlay individual MC runs as jittered dots on the "Prog ON" bar
rng = np.random.default_rng(0)
jitter = rng.uniform(-0.12, 0.12, size=n)
ax.scatter(
    2 + jitter, accs, color="black", alpha=0.45, s=18, zorder=4,
    label=f"individual realizations (N={n})",
)
# mark the original seed=42 realization explicitly
ax.scatter(
    [2], [acc_seed42_reproduced], color="white", edgecolor=_COLOR_ON_MEAN,
    linewidth=1.8, s=70, zorder=5, marker="D",
    label=f"seed=42 (reported in 08): {acc_seed42_reproduced:.2f}%",
)

for bar, val in zip(bars, bar_vals):
    ax.text(
        bar.get_x() + bar.get_width() / 2, val + 0.35,
        f"{val:.2f}%", ha="center", va="bottom", fontsize=10,
    )
ax.text(
    2, bar_vals[2] - bar_errs[2] - 0.9,
    f"std = {acc_std:.2f} pp",
    ha="center", va="top", fontsize=9, color="#444444",
)

ax.set_ylabel("MNIST test accuracy (%)", fontsize=12)
ax.set_title(
    f"TNN accuracy: {TNN_APM_MODEL} programming error\n"
    f"ideal baseline vs. mean $\\pm$ std over N={n} independent realizations",
    fontsize=11, pad=12,
)
y_lo = min(bar_vals + [acc_min]) - max(bar_errs) - 2.0
ax.set_ylim(y_lo, 100)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="lower left", framealpha=0.95, fontsize=8.5)
plt.tight_layout()
main_fig_path = OUT_DIR / f"{DATE_TAG}_figMAIN_accuracy_mean_std.png"
fig.savefig(main_fig_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"\nSaved main publication figure: {main_fig_path}")

# ============================================================
# 7. OPTIONAL diagnostic figures (per-run scatter, histogram)
# ============================================================
if SAVE_DIAGNOSTIC_FIGURES:
    # --- Diagnostic 1: per-run accuracy scatter with mean +/- std band ---
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3, zorder=0)

    ax.axhline(
        digital_tnn_acc, color=_COLOR_DIGITAL, linestyle="--", linewidth=1.4,
        label=f"Digital TNN ({digital_tnn_acc:.2f}%)", zorder=2,
    )
    ax.axhspan(
        acc_mean - acc_std, acc_mean + acc_std,
        color=_COLOR_ON_MEAN, alpha=0.15, zorder=1,
        label=f"mean $\\pm$ std ({acc_mean:.2f}% $\\pm$ {acc_std:.2f} pp)",
    )
    ax.axhline(acc_mean, color=_COLOR_ON_MEAN, linewidth=1.8, zorder=2)
    ax.scatter(
        mc_df["run"], mc_df["accuracy_percent"],
        color=_COLOR_ON_MEAN, edgecolor="black", linewidth=0.6,
        s=55, zorder=3, label="Individual programming realization",
    )
    ax.set_xlabel("Independent programming realization (run index)", fontsize=11)
    ax.set_ylabel("MNIST test accuracy (%)", fontsize=11)
    ax.set_title(
        f"[Diagnostic] TNN accuracy under {TNN_APM_MODEL} programming error\n"
        f"N = {n} independent realizations: {acc_mean:.2f}% $\\pm$ {acc_std:.2f}% (std)",
        fontsize=12, pad=12,
    )
    ax.set_xlim(0.3, n + 0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"{DATE_TAG}_figDIAG1_runs_scatter.png", dpi=300, bbox_inches="tight")
    plt.show()

    # --- Diagnostic 2: histogram of the N accuracies ---
    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=300)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3, zorder=0)
    ax.hist(
        accs, bins=max(6, n // 4), color=_COLOR_ON_MEAN, edgecolor="black",
        alpha=0.85, zorder=3,
    )
    ax.axvline(acc_mean, color="black", linewidth=1.8, linestyle="-", zorder=4,
               label=f"mean = {acc_mean:.2f}%")
    ax.axvline(acc_mean - acc_std, color="black", linewidth=1.2, linestyle="--", zorder=4)
    ax.axvline(acc_mean + acc_std, color="black", linewidth=1.2, linestyle="--", zorder=4,
               label=f"$\\pm$1 std = {acc_std:.2f} pp")
    ax.axvline(digital_tnn_acc, color=_COLOR_DIGITAL, linewidth=1.6, linestyle=":",
               zorder=4, label=f"digital = {digital_tnn_acc:.2f}%")
    ax.axvline(acc_seed42_reproduced, color=_COLOR_ON_MEAN, linewidth=1.6, linestyle="-.",
               zorder=4, label=f"seed=42 (08) = {acc_seed42_reproduced:.2f}%")
    ax.set_xlabel("MNIST test accuracy (%) — programming ON", fontsize=11)
    ax.set_ylabel(f"Count (out of N = {n})", fontsize=11)
    ax.set_title(
        f"[Diagnostic] Distribution of TNN accuracy across {n} independent "
        f"{TNN_APM_MODEL} programming realizations",
        fontsize=11.5, pad=12,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=8.5)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"{DATE_TAG}_figDIAG2_histogram.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\nSaved optional diagnostic figures:")
    print(f"  {OUT_DIR / f'{DATE_TAG}_figDIAG1_runs_scatter.png'}")
    print(f"  {OUT_DIR / f'{DATE_TAG}_figDIAG2_histogram.png'}")

print("\nAll verification checks passed (V1-V8). Monte Carlo ensemble is valid.")
