# MIDRC-MELODY (Model EvaLuation across subgroups for cOnsistent Decision accuracY)

[Overview](#overview) | [Requirements](#requirements) | [Setup](#setup) | [Usage](#usage) | [License](#license)

[📱 Visit MIDRC Website](https://www.midrc.org/)

**MIDRC-MELODY** is a tool designed to assess the performance and subgroup-level reliability and robustness of AI models
developed for medical imaging analysis tasks, such as the estimation of disease severity. It enables consistent
evaluation of models across predefined subgroups (e.g. manufacturer, race, scanner type) by computing intergroup
performance metrics and corresponding confidence intervals.

The tool supports two types of evaluation:
- **Ordinal Estimation Task Evaluation**:
  - Uses an ordinal reference standard ("truth") and ordinal AI model outputs.
  - Performance in terms of agreement of AI output with the reference standard "truth" is quantified using the **quadratic
    weighted kappa (QWK)** metric.
  - Subgroup reliability is assessed using the **delta QWK** metric, which quantifies the difference in QWK between a reference
    subgroup and other subgroups.
  - A delta QWK whose 95% confidence interval does not include 0 indicates a statistically significant difference in performance
    between the reference subgroup and the other subgroup.
- **Binary Classification Task Evaluation**:
  - The ordinal "truth" values are binarized into two classes (e.g. "positive" and "negative") based on a user-defined threshold.
  - The same ordinal AI model outputs are used, but the evaluation is performed as a binary classification task.
  - Reliability across subgroups is assessed using:
    - **Equal Opportunity Difference (EOD)** metric, which quantifies the difference in true positive rates between a reference
      subgroup and other subgroups.
    - **Average Absolute Odds Difference (AAOD)** metric, which is the average of the absolute differences in true positive
      rates and false positive rates between a reference subgroup and other subgroups.
  - An EOD *outside* the range [-0.1, 0.1] and AAOD outside the range [0.0, 0.1] are typically defined as indicating
    an observable discrepancy in model performance across subgroups.


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
   - On macOS/Linux: `source venv/bin/activate`
4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Configuration

The project uses a YAML configuration file (`config.yaml`) to specify:
- Input data and column names (truth column, numeric columns, etc.)
- Bootstrap settings (seed and iterations)
- Plot settings and output paths

Ensure to update the configuration file according to your dataset and desired parameters.

### Input Data
MIDRC-MELODY requires two CSV input files:
- **Model predictions**: It must include the following columns:
  - case_name: Unique identifier for each case.
  - One or more columns each containing a model's ordinal predictions.
- **Reference standard**: It must include the following columns:
  - case_name: Unique identifier for each case (must match the model predictions file).
  - truth: The ordinal reference standard values (e.g. 0-4).
  - One or more subgroup columns (e.g. manufacturer, race, scanner_type), each with categorical values for stratification.
  - Note:
    - The largest subgroup for a given category is used as the reference group for delta QWK calculations.
    - Subgroups with too few cases (less than 10 by default) are excluded from the analysis.

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

### Data Processing and Visualization

- **Bootstrapping:** Both scripts perform bootstrapping to compute confidence intervals for the respective metrics using NumPy's percentile method.
- **Plotting:** Spider charts provide a visual overview of how each model's metrics vary across different groups and categories.
- **Utilities:** Shared functionality is available in common utility modules (e.g., `data_loading.py` and `plot_tools.py`), ensuring easier maintenance and testing.

## License

Apache License 2.0

