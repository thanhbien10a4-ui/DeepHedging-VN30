"""Capture the exact runtime environment used by the final project.

This script does not install, upgrade, or modify any package. It only records
versions and basic runtime information so requirements.txt can be pinned from
the environment that actually produced the project results.
"""

from __future__ import annotations

import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path

PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "matplotlib",
    "jupyter",
    "PyYAML",
    "torch",
]


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    packages = {name: package_version(name) for name in PACKAGES}

    torch_info: dict[str, object] = {}
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "num_threads": int(torch.get_num_threads()),
            "num_interop_threads": int(torch.get_num_interop_threads()),
            "device_count": int(torch.cuda.device_count()),
        }
        if torch.cuda.is_available():
            torch_info["device_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except Exception as exc:  # diagnostic only
        torch_info = {"error": repr(exc)}

    snapshot = {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": packages,
        "torch_runtime": torch_info,
    }

    out_path = out_dir / "environment_snapshot.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    print("=" * 96)
    print("FINAL PROJECT ENVIRONMENT SNAPSHOT")
    print("=" * 96)
    print(f"Python         : {snapshot['python']['version']}")
    print(f"Implementation : {snapshot['python']['implementation']}")
    print(f"Executable     : {snapshot['python']['executable']}")
    print(f"OS             : {snapshot['platform']['system']} {snapshot['platform']['release']}")
    print(f"Machine        : {snapshot['platform']['machine']}")
    print()
    print("PACKAGE VERSIONS")
    for name in PACKAGES:
        version = packages[name]
        print(f"{name:14s}: {version if version is not None else 'NOT INSTALLED'}")
    print()
    print("TORCH RUNTIME")
    if "error" in torch_info:
        print(f"Torch error    : {torch_info['error']}")
    else:
        print(f"torch          : {torch_info['version']}")
        print(f"CUDA available : {torch_info['cuda_available']}")
        print(f"CUDA version   : {torch_info['cuda_version']}")
        print(f"cuDNN version  : {torch_info['cudnn_version']}")
        print(f"CPU threads    : {torch_info['num_threads']}")
        print(f"Interop threads: {torch_info['num_interop_threads']}")
    print()
    print(f"Snapshot saved : {out_path}")
    print("No package was installed, upgraded, or modified.")


if __name__ == "__main__":
    main()
