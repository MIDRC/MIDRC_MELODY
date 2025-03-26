# MIDRC AIRWD Spider Plot Generator

This project calculates and visualizes bootstrapped metrics for multiple AI models. It performs two main tasks:  
- Calculation of Quadratic Weighted Kappa (QWK) differences, with bootstrapping and spider plot visualizations.  
- Calculation of EOD (Equal Opportunity Difference) and AAOD (Average Absolute Opportunity Difference) metrics with bootstrap confidence intervals and corresponding spider plots.

## Overview

The project consists of two main scripts:
- `generate_qwk_spiders.py`: Computes delta QWK values for different groups, applies bootstrapping for confidence intervals and generates spider plot visualizations.
- `generate_eod_aaod_spiders.py`: Calculates bootstrapped estimates for EOD and AAOD metrics across multiple categories and generates spider charts to visualize these measures.

Additional modules such as `data_loading.py` and `plot_tools.py` provide functions to load data, check required columns, and generate plots. Common functionalities are consolidated into shared utility functions to adhere to DRY principles.

## Requirements

- Python 3.x
- pip

### Python Packages

- matplotlib
- numpy
- pandas
- joblib
- scikit-learn
- tqdm
- tqdm\_joblib
- pyyaml
- tabulate

## Setup

1. Clone the repository to your local machine.
2. Create a virtual environment (optional recommended):
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The project uses a YAML configuration file (`config.yaml`) to specify:
- Input data and column names (truth column, numeric columns, etc.)
- Bootstrap settings (seed and iterations)
- Plot settings and output paths

Ensure to update the configuration file according to your dataset and desired parameters.

## Usage

### Generate QWK Spider Plots

Run the following command to execute the QWK spider plot script:
```bash
python generate_qwk_spiders.py
```
This script:
- Loads the dataset.
- Checks for required columns.
- Calculates Cohen's quadratic weighted kappa and bootstrapped confidence intervals.
- Computes delta kappa values comparing each group against a reference.
- Generates and displays spider plots.
- Saves the results to a pickle file.

### Generate EOD & AAOD Spider Plots

Run the following command to execute the EOD and AAOD spider plot script:
```bash
python generate_eod_aaod_spiders.py
```
This script:
- Loads and preprocesses the dataset (including score binarization based on a threshold).
- Computes EOD and AAOD metrics using bootstrapping across various groups.
- Generates spider plots comparing these metrics.
- Saves the generated data for further analysis.

## Data Processing and Visualization

- **Bootstrapping:** Both scripts perform bootstrapping to compute confidence intervals for the respective metrics using NumPy's percentile method.
- **Plotting:** Spider charts provide a visual overview of how each model's metrics vary across different groups and categories.
- **Utilities:** Shared functionality is available in common utility modules (e.g., `data_loading.py` and `plot_tools.py`), ensuring easier maintenance and testing.

## License

Apache License 2.0

