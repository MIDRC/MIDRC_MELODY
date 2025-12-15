# MIDRC MELODY (Model EvaLuation across subgroups for cOnsistent Decision accuracY)

[Overview](#overview) | [Installation](#installation) | [CLI Commands](#cli-commands) | [Configuration](#configuration) | [License](#license)

[📱 Visit MIDRC Website](https://www.midrc.org/)

**MIDRC MELODY** is a tool designed to assess the performance and subgroup-level reliability and robustness of AI models
developed for medical imaging analysis tasks, such as the estimation of disease severity. It enables consistent
evaluation of models across predefined subgroups (e.g. manufacturer, race, scanner type) by computing intergroup
performance metrics and corresponding confidence intervals.

The tool supports two types of evaluation:

- **Ordinal Estimation Task Evaluation**:
  - Uses an ordinal reference standard ("truth") and ordinal AI model outputs.
  - Performance in terms of agreement of AI output with the reference standard "truth" is quantified using the **quadratic
    weighted kappa (QWK)** metric.
- **Binary Decision Task Evaluation**:
  - Converts scores into binary decisions based on a threshold.
  - Computes **Equal Opportunity Difference (EOD)** and **Average Absolute Odds Difference (AAOD)** metrics using bootstrapping across various groups.
- Generates spider plots comparing these metrics.
- Saves the generated data for further analysis.

### Data Processing and Visualization

- **Bootstrapping:** Both scripts perform bootstrapping to compute confidence intervals for the respective metrics using NumPy's percentile method.
- **Plotting:** Spider charts provide a visual overview of how each model's metrics vary across different groups and categories.
- **Utilities:** Shared functionality is available in common utility modules (e.g., `data_tools.py` and `plot_tools.py`), ensuring easier maintenance and testing.

## Overview

**MIDRC MELODY** is a lightweight toolkit for stress‑testing medical‑imaging AI models across clinical and demographic sub‑groups. It supports both command‑line and GUI workflows, enabling rapid quantification of performance disparities (QWK, EOD, AAOD, etc.) and intuitive radar‑chart visualisation.

- **Console‑first** – core metrics and plots run with **no GUI dependencies**.
- **Opt‑in GUI** – an optional PySide6 interface for interactive configuration and result browsing.
- **Config‑driven** – YAML files keep experiments reproducible and shareable.

## Installation

```bash
# Standard console install from PyPI
pip install midrc-melody

# (Alternative) Install in editable/development mode from source code
pip install -e .
```

## Quick Start

```bash
# Run analysis (reads default config.yaml in current directory)
melody

# Launch the GUI
melody_gui
```

## CLI Commands

Running `melody` opens a **Command‑Line Interface (CLI)**, which presents a text‑based menu of interactive commands. Here’s what you can do:

#### Available Commands

- **Calculate QWK metrics**: Computes delta QWK values for different subgroups and generates spider plots.
- **Calculate EOD and AAOD metrics**: Computes EOD and AAOD metrics for binary decision tasks and generates spider plots.
- **Print config file contents**: Displays the contents of the current YAML configuration file.
- **Change config file**: Prompts you to enter and set a different configuration file path.
- **Launch GUI**: Opens the Graphical User Interface (GUI) using PySide6 (requires PySide6).
- **Exit**: Exits the program.

## GUI (Optional)

Launching the graphical interface only requires that PySide6 is installed. If you used pip, the `melody_gui` command is available.

```bash
# Launch the GUI:
melody_gui
```

## Configuration

Experiments are described in a single YAML file. Below is a **minimal** example that keeps storage light and avoids plotting custom order metadata.

```yaml
input data:
  truth file: data/demo_truthNdemographics.csv
  test scores: data/demo_scores.csv
  uid column: case_name
  truth column: truth

# Scores ≥ binary threshold are counted as positive
binary threshold: 4
min count per category: 10

bootstrap:
  iterations: 1000
  seed: 42  # set to null for random entropy

output:
  qwk:  { save: false, file prefix: output/delta_kappas_ }
  eod:  { save: false, file prefix: output/eod_ }
  aaod: { save: false, file prefix: output/aaod_ }

numeric_cols:
  age_binned:
    raw column: age
    bins: [0, 18, 30, 40, 50, 65, 75, 85, .inf]

plot:
  clockwise: true            # rotate clockwise instead of CCW
  start: top                 # starting angle: top, bottom, left, right (t/b/l/r)
```

## Input Data

| File           | Required Columns            | Purpose                                   | Example                                            |
| -------------- | --------------------------- | ----------------------------------------- |----------------------------------------------------|
| **Truth file** | `uid`, `truth`, attributes… | Ground‑truth labels and subgroup columns. | [demo_truth.csv](data/demo_truthNdemographics.csv) |
| **Score file** | `uid`, `score`              | Model predictions keyed to the same UID.  | [demo_scores.csv](data/demo_scores.csv)            |

> UID values must match between truth and score files.

## License

Distributed under the Apache 2.0 License.

