# Models Carbon Emissions

A collection of Python experiments measuring carbon emissions and energy consumption of various computational tasks using [CodeCarbon](https://codecarbon.io/).

## Overview

This project demonstrates how to track and compare CO2 emissions across different algorithms, ML models, and geographic locations. It uses CodeCarbon to measure the environmental impact of code execution.

## Experiments

| File | Description |
|------|-------------|
| `ex1_codecarbon.py` | Basic CodeCarbon usage - tracks emissions while computing factorials |
| `ex2_sort_compare.py` | Compares emissions of sorting algorithms (Insertion Sort vs Merge Sort) |
| `ex3_train_compare.py` | Compares CPU vs GPU training emissions for a PyTorch MLP |
| `ex4_location_compare.py` | Compares emissions based on geographic location (FRA, USA, CHN, SWE) |
| `ex6_report_visualization.py` | Generates emission reports and visualizations |
| `ex7_models_compare.py` | Compares emissions across ML models (LogReg, SVM, RandomForest, MLP) |

## Requirements

- Python 3.8+
- codecarbon
- numpy
- pandas
- matplotlib
- scikit-learn
- torch (for ex3)

## Installation

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install codecarbon numpy pandas matplotlib scikit-learn torch
```

## Usage

Run any experiment individually:

```bash
python ex1_codecarbon.py
python ex2_sort_compare.py
python ex3_train_compare.py
python ex4_location_compare.py
python ex6_report_visualization.py
python ex7_models_compare.py
```

## Output

- `emissions.csv` - Raw emissions data logged by CodeCarbon
- `reports/` - Contains aggregated results and visualizations:
  - `emissions.csv` - Emissions data from experiments
  - `ex7_results.csv` - Model comparison results
  - `*.png` - Generated charts and plots

## Key Insights

- **Algorithm efficiency matters**: More efficient algorithms produce fewer emissions
- **Location impacts emissions**: Carbon intensity varies by country's energy grid
- **Model complexity vs emissions**: Simpler models often have lower carbon footprint
- **GPU vs CPU**: Trade-offs between speed and energy consumption

## License

Abdelaziz LOUNES / @Paris8
