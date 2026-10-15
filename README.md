# Scalable Identification of Physics-Based and Neural Dynamics from Partial Measurements
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains code for the paper:
>  Yibo Zhou, Johannes Köhler "Scalable Identification of Physics-Based and Neural Dynamics from Partial Measurements", preprint available on **arXiv**: [[Link]](https://arxiv.org/abs/2610.16559)

## Installation

Use Python 3.10.

1. Clone the repository:

   ```sh
   git clone https://github.com/YiboFrankZhou/Scalable-PEM-MHE.git
   cd Scalable-PEM-MHE
   ```

2. Install the Python dependencies:

   ```sh
   python -m pip install -r requirements.txt
   ```

3. Install [acados](https://docs.acados.org/installation/index.html) and its [Python interface](https://docs.acados.org/python_interface/index.html#installation) following the official instructions.

## Reproducing the experimental results

The following notebooks reproduce the paper's three numerical experiments:

| Paper section | Notebooks |
|---|---|
| **4.2 Computational analysis** | [Ex1_gradient_computation_acceleration.ipynb](Ex1_gradient_computation_acceleration.ipynb)<br>[Ex1_plot.ipynb](Ex1_plot.ipynb) |
| **4.3 Simulation: Parameter Recovery and Prediction** | [Ex2_SUBNET.ipynb](Ex2_SUBNET.ipynb)<br>[Ex2_Joint-MHE.ipynb](Ex2_Joint-MHE.ipynb)<br>[Ex2_EKF-PEM.ipynb](Ex2_EKF-PEM.ipynb)<br>[Ex2_EKF-PEM-adaptive.ipynb](Ex2_EKF-PEM-adaptive.ipynb)<br>[Ex2_PEM-MHE.ipynb](Ex2_PEM-MHE.ipynb) |
| **4.4 Real-World Data Validation** | [Ex3_SUBNET.ipynb](Ex3_SUBNET.ipynb)<br>[Ex3_Joint-MHE.ipynb](Ex3_Joint-MHE.ipynb)<br>[Ex3_EKF-PEM.ipynb](Ex3_EKF-PEM.ipynb)<br>[Ex3_EKF-PEM-adaptive.ipynb](Ex3_EKF-PEM-adaptive.ipynb)<br>[Ex3_PEM-MHE-physics.ipynb](Ex3_PEM-MHE-physics.ipynb)<br>[Ex3_PEM-MHE-graybox.ipynb](Ex3_PEM-MHE-graybox.ipynb) |

The notebooks call six Python modules in the repository root:

| File | Purpose |
|---|---|
| [models.py](models.py) | Dynamics, physical parameters, neural networks, and discretization |
| [data.py](data.py) | Data loading, preprocessing, and observation windows |
| [estimation.py](estimation.py) | MHE configuration and solver, EKF/RTS state estimation, and initial trajectories |
| [identification.py](identification.py) | PEM-MHE framework (Section 3.1) and comparison methods |
| [gradients.py](gradients.py) | KKT blocks, block adjoint solve, and hypergradients (Section 3.2) |
| [utils.py](utils.py) | Runtime setup, parameter saving/loading, model display, and timing |

The repository includes the prepared data in `datasets/`, pretrained parameters and training records in `runs/`, and third-party implementations in `repos/` (see [dependencies and licenses](repos/README.md)).

After completing [Installation](#installation), start Jupyter Notebook from the repository root:

```sh
jupyter notebook
```

Open the corresponding notebook, set the flags in its configuration cell as described below, and run all cells.

### Ex1

- In `Ex1_gradient_computation_acceleration.ipynb`, keep `RUN_BENCHMARK = False` to print the saved timings, or set it to `True` to rerun the benchmark and save outputs to `runs/new/Ex1_<timestamp>/`.
- Then run `Ex1_plot.ipynb` to save the images alongside the timing data in `runs/Ex1/`.

### Ex2/Ex3

- **View training curves.** Run `tensorboard --logdir runs` from the repository root to inspect the saved training histories.
- **Print results using pretrained parameters.** In any Ex2/Ex3 notebook, keep `LOAD_PRETRAINED = True` (the default) to load the saved parameters and print the model and computed training/validation PEM costs. *(Note that MHE state estimates depend on the acados solver’s warm start, so results may differ slightly from those reported in the paper.)*
- **Train a model from scratch.** In the corresponding Ex2/Ex3 notebook, set `LOAD_PRETRAINED = False` and run all cells. New training outputs are saved to `runs/new/<experiment>_<timestamp>/`.

## Applying the framework to other systems

Copy [Ex2_PEM-MHE.ipynb](Ex2_PEM-MHE.ipynb) to a new notebook in the repository root. Make the system-specific changes in this new notebook.

- **Configure the system — Configuration cell.** Set `nx`, `nu`, `nw`, horizon `N`, sample time, measurement matrix `C`, process-noise matrix `E`, initial parameters, and arrival/noise weights. Set `LOAD_PRETRAINED = False`.
- **Define the dynamics — Data and model cell; optionally `models.py`.** Replace `make_vehicle(...)` with your own parameterized dynamics function. Provide matching PyTorch and CasADi dynamics with the same parameter ordering and discretization. Supply your own initial parameters with `load_initialization(NAME)`.
- **Prepare the data — Data and model cell; optionally `data.py`.** Replace `load_data(EXPERIMENT)` with your own loader. Provide `data[split]["u"]` with shape `(windows, N, nu)` and `data[split]["y"]` with shape `(windows, N, ny)` for `train` and `val`.
- **Set up state estimation — Inner MHE cell.** Replace the vehicle-specific `make_mhe_config(...)` call with a `MHEConfig` instance imported from `estimation.py`, supplying your dynamics, dimensions, weights, symbolic parameters (`theta`), and parameter-value callback (`theta_values`). Replace the vehicle-specific EKF/grid initialization with initial state trajectories of shape `(windows, N + 1, nx)`, passed to `problem.estimate(split, warmstart=...)`. The generic `ObserverMHEAcados` solver in `estimation.py` can be reused.
- **Run identification — parameter-update and results cells.** Create `PEMMHE(model, config, data)` using the class in `identification.py`. Keep the `problem.estimate(split)` / `problem.backward()` workflow, but adapt the optimizer parameter groups, checkpoint saving/loading, and result display to your model instead of using the vehicle-specific helpers. The core `PEMMHE.estimate` / `PEMMHE.backward` methods and `gradients.py` can be reused for the formulation above.
