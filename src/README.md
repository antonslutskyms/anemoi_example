`src/submit_training_aml.sh` must be copied to /<YOUR PATH TO>/replay/atmosphere-subsampled/p0/training

---

This repository is a minimal setup for running Anemoi training workloads. The project structure has two main directories:

- `environment/` – contains the Dockerfile and a Conda environment configuration.
  - The environment file lists dependencies like Python 3.11, NumPy, various scientific libraries, and packages related to the Anemoi ecosystem (training, models, graphs, datasets, and utilities).
  - The Dockerfile uses a Microsoft AzureML PyTorch base image and installs the environment with `conda env update`.

- `src/` – holds a single shell script for submitting Anemoi training to Azure ML.
  - The script expects a data path and output directory, sets up symlinks within a scratch directory, exports several environment variables (e.g., `WORLD_SIZE`, `SLURM_GPUS_PER_NODE`), installs `flash-attn`, and invokes `anemoi-training train --config-name=config`.
  - The accompanying README simply notes that the script should be copied to a specific "replay" path before use.

The repository's single commit sets up this basic structure, providing a starting point for Anemoi experiments with containers and training scripts.
