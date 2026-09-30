"""Folders and checkpoint picking for this project.

Paths are relative to the repository (the folder that contains this file),
so the scripts run on any machine. Set an environment variable only when a
folder lives somewhere else:

    PTH_DIR          saved model files     default: <repo>/Pth_Models
    DEVICE_DATA_DIR  drift CSVs and I-V    default: <repo>/Device_Data
    MNIST_DIR        torchvision MNIST     default: <repo>/data
    CROSS_SIM_DIR    CrossSim checkout     default: <repo>/cross-sim, then ../cross-sim

Load scripts ask which file to use when more than one match is in Pth_Models.
Skip the question by setting CHECKPOINT_ANN, CHECKPOINT_BNN, CHECKPOINT_TNN,
CHECKPOINT_TNN_TRAIN, CHECKPOINT_DIGITAL_TNN, or CHECKPOINT_DEVICE_AWARE_TNN
to a full path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

_ROLE_SPECS = {
    "ann": {
        "label": "full-precision ANN",
        "prefixes": ("BASE_ANN_", "mnist_ann_"),
    },
    "bnn": {
        "label": "binary BNN",
        "name_contains": "_BNN",
    },
    "tnn": {
        "label": "ternary-only TNN (inference weights -1, 0, +1)",
        "prefixes": ("TERNARY_ONLY_",),
    },
    "tnn_train": {
        "label": "trained TNN (latent weights)",
        "prefixes": ("BASE_TNN_",),
    },
    "digital_tnn": {
        "label": "digital TNN from device-aware training",
        "name_contains": "digital_tnn",
    },
    "device_aware_tnn": {
        "label": "device-aware TNN",
        "name_contains": "device_aware",
    },
}

_MODEL_SUFFIXES = (".pth", ".pt")


def _resolve_dir(env_name: str, default: Path) -> Path:
    raw = os.environ.get(env_name)
    path = Path(raw).expanduser().resolve() if raw else default
    path.mkdir(parents=True, exist_ok=True)
    return path


PTH_DIR = _resolve_dir("PTH_DIR", PROJECT_DIR / "Pth_Models")
DEVICE_DATA_DIR = _resolve_dir("DEVICE_DATA_DIR", PROJECT_DIR / "Device_Data")
DATA_DIR = _resolve_dir("MNIST_DIR", PROJECT_DIR / "data")
FIGURES_DIR = PROJECT_DIR / "figures"


def cross_sim_dir() -> Path:
    """Return the CrossSim checkout, or raise with the places that were checked."""
    raw = os.environ.get("CROSS_SIM_DIR")
    candidates = []
    if raw:
        candidates.append(Path(raw).expanduser())
    candidates.extend(
        [
            PROJECT_DIR / "cross-sim",
            PROJECT_DIR.parent / "cross-sim",
        ]
    )
    for candidate in candidates:
        if (candidate / "simulator").is_dir():
            return candidate.resolve()
    looked = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(
        "CrossSim was not found. Put the checkout in cross-sim/ next to these\n"
        "scripts, or set CROSS_SIM_DIR to that folder.\n"
        f"Looked in:\n{looked}"
    )


def ensure_cross_sim_on_path() -> Path:
    folder = cross_sim_dir()
    folder_str = str(folder)
    if folder_str not in sys.path:
        sys.path.insert(0, folder_str)
    return folder


def list_checkpoints(role: str, folder: Path | None = None) -> list[Path]:
    """Model files in ``folder`` that belong to ``role``, newest first."""
    if role not in _ROLE_SPECS:
        known = ", ".join(sorted(_ROLE_SPECS))
        raise KeyError(f"Unknown checkpoint role {role!r}. Use one of: {known}")

    root = folder or PTH_DIR
    spec = _ROLE_SPECS[role]
    prefixes = spec.get("prefixes") or ()
    needle = spec.get("name_contains")
    matches = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in _MODEL_SUFFIXES:
            continue
        name = path.name
        if prefixes and name.startswith(prefixes):
            matches.append(path)
        elif needle and needle in name:
            matches.append(path)
    matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return matches


def choose_checkpoint(role: str, folder: Path | None = None) -> Path:
    """Pick the checkpoint for ``role``.

    One match is used as-is. Several matches are printed and the script asks
    for a number (Enter keeps the newest). A non-interactive run uses the
    newest file so it does not wait on input.
    """
    if role not in _ROLE_SPECS:
        known = ", ".join(sorted(_ROLE_SPECS))
        raise KeyError(f"Unknown checkpoint role {role!r}. Use one of: {known}")

    label = _ROLE_SPECS[role]["label"]
    env_key = "CHECKPOINT_" + role.upper()
    override = os.environ.get(env_key)
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{env_key} is not a file: {path}")
        print(f"{label}: {path}  (from {env_key})")
        return path.resolve()

    root = folder or PTH_DIR
    matches = list_checkpoints(role, root)
    if not matches:
        raise FileNotFoundError(
            f"No {label} file in {root}.\n"
            "Train the script that saves this model, copy a .pth/.pt into that\n"
            f"folder, or set {env_key} to the file path."
        )

    if len(matches) == 1 or not sys.stdin.isatty():
        chosen = matches[0]
        note = "" if len(matches) == 1 else "  (newest; no terminal prompt)"
        print(f"{label}: {chosen.name}{note}")
        return chosen

    print(f"\nModels folder: {root}")
    print(f"Choose {label}:")
    for index, path in enumerate(matches, start=1):
        try:
            shown = path.relative_to(root)
        except ValueError:
            shown = path
        print(f"  [{index}] {shown}")

    while True:
        raw = input(f"Number 1-{len(matches)} (Enter = 1, newest): ").strip()
        if raw == "":
            return matches[0]
        if raw.isdigit() and 1 <= int(raw) <= len(matches):
            return matches[int(raw) - 1]
        print("Type a number from the list.")
