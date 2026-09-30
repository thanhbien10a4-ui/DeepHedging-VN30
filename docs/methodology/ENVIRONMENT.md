# Final Runtime Environment

This document records the environment used for the final DeepHedging-VN30 experimental pipeline.

## Platform

- Operating system: Windows 11
- Architecture: AMD64
- Python implementation: CPython
- Python version: 3.13.4
- Deep-learning execution: CPU only
- CUDA available: No

## Pinned runtime packages

| Package | Version |
|---|---:|
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| scipy | 1.16.2 |
| scikit-learn | 1.8.0 |
| statsmodels | 0.14.6 |
| matplotlib | 3.10.3 |
| PyYAML | 6.0.3 |
| torch | 2.13.0 |

The PyTorch runtime identified itself as `2.13.0+cpu`. The Python package metadata version is `2.13.0`, so `requirements.txt` pins `torch==2.13.0` while documenting that the final run was CPU-only.

## Notebook interface

`jupyter` was not installed in the final environment that produced the reported project results. It is therefore not a required dependency in `requirements.txt`. It may be installed separately for interactive notebook work without being treated as part of the final reproducibility environment.

## Reproducibility policy

The exact observed package versions are pinned rather than using unbounded dependencies. Generated processed/synthetic data remain excluded from Git tracking, so reproduction requires rerunning the preprocessing, simulator, model-training, and evaluation scripts from the tracked raw data and source code.

The environment snapshot can be regenerated locally with:

```powershell
python src\utils\29_capture_environment.py
```

This command only reads the environment; it does not install or modify packages.

## Final-evaluation integrity

The real TEST split was opened only after the model was frozen. Environment pinning and documentation were performed afterward and do not alter the trained checkpoints, model architecture, feature schema, hyperparameters, or final TEST results.
