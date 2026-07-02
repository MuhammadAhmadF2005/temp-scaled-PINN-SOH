# Physics-Informed Neural Networks for Battery State of Health Estimation with Arrhenius Scaling

## Overview
This repository contains the implementation of a Physics-Informed Neural Network (PINN) framework for the State of Health (SOH) estimation of lithium-ion batteries. 

Standard deep learning models for battery lifetime prognostics often treat cycle numbers as a raw, linear time index, failing to capture the acceleration of aging kinetics at elevated temperatures. This project addresses this limitation by introducing a temperature-scaled cycle time utilizing the Arrhenius equation. This transformation is embedded directly into the forward pass of the neural network as a learnable activation energy parameter ($E_a$).

Additionally, the dataset loader incorporates a rolling median smoothing filter to eliminate measurement anomalies (e.g., transient drops during reference performance tests or diagnostic cycles), yielding smooth physical degradation trajectories and improving model training stability.

## Features
- **Arrhenius Temporal Scaling:** Adapts cycle numbers to an equivalent "thermal age" based on a learnable activation energy parameter.
- **Physics-Informed Optimization:** Integrates differential aging equations and monotonicity constraints into the loss function to guide predictions.
- **Robust Data Cleaning:** Implements a rolling median filter (window size 5) to remove transient capacity drops and diagnostic cycle anomalies.
- **Cross-Temperature Generalization:** Demonstrates high generalization performance when tested on unseen temperature profiles (e.g., cycled at 35°C, trained on 25°C and 45°C).

---

## File Structure
- `data_loader.py`: Dataset loader, feature extractor, and rolling median trajectory smoothing utility.
- `model.py`: Definition of the PINN network architecture (`BatteryPINN`) and physics loss calculation.
- `train.py`: Model training pipeline, implementation of early stopping, and test set evaluator.
- `visualize.py`: Script to generate predictions and plot SOH trajectories against true SOH.
- `runner.sh`: Shell script to execute all holdout and random split benchmark experiments.
- `Dataset_1_NCA_battery/`: Subfolder containing raw cycling CSV files from the NCA battery cycling dataset.
- `plots/`: Directory containing generated visualization plots comparing baseline and scaling configurations.
- `plots.zip`: Archive containing the completed visualization outputs.

---

## Installation and Setup

### Prerequisites
Ensure Python 3.8+ is installed along with the required dependencies:
```bash
pip install torch numpy pandas scipy matplotlib
```

### Dataset
Place all NCA battery CSV data files in the `Dataset_1_NCA_battery` folder in the project root.

---

## Usage

### 1. Run Benchmarks
You can execute all configurations (holdout test at 35°C and random splits, for both scaled and unscaled models) by running the runner script:
```bash
# On Unix-like systems
./runner.sh
```

### 2. Manual Training
To train the scaled model on a specific holdout temperature (e.g., 35°C):
```bash
python train.py --holdout_temp 35.0 --plot_dir plots_holdout_scaled
```

To train a baseline model (without Arrhenius scaling) on the same holdout temperature:
```bash
python train.py --no-scaling --holdout_temp 35.0 --plot_dir plots_holdout_unscaled
```

To run a random split train-test partition:
```bash
python train.py --plot_dir plots_random_scaled
```

---

## Experimental Results
Models were evaluated on a holdout test set cycled at 35°C (completely unseen during training). The table below showcases the performance improvements gained by integrating Arrhenius scaling and rolling median data smoothing:

### Holdout Test Set (35°C Unseen)
| Configuration | MSE ($mA^2 \cdot h^2$) | MAE ($mA \cdot h$) | RMSE ($mA \cdot h$) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (No Scaling)** | 1123.71 | 26.40 | 33.52 | 0.471 |
| **Proposed (Arrhenius Scaling)** | **254.64** | **13.83** | **15.96** | **0.880** |

Integrating Arrhenius temporal scaling reduced holdout test MSE by **77.3%** and raised the $R^2$ score to **0.880**, validating the physical model's ability to extrapolate degradation across temperatures.
