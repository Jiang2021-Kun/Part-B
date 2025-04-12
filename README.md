# Part-A

# RFL/FSO Channel Attenuation Prediction

This repository contains the code implementation for channel attenuation prediction in hybrid FSO/RF communication systems.

## Main Files

1. `feature_selection.py`
   - Implements parallel feature selection algorithm
   - Processes different weather conditions simultaneously
   - Generates feature importance rankings for both RFL and FSO systems

2. `code_final.ipynb`
   - Contains model training and evaluation
   - Implements both general and weather-specific models
   - Generates performance comparison visualizations
   - Includes hyperparameter tuning

## Usage

⚠️ **Note**: Feature selection and model training are computationally intensive. Expected runtime:
- Feature selection: ~40 minutes per system on a 16-core CPU
- jupyter notebook: ~2 hour for all weather conditions


1. First run feature selection:

```shell
python feature_selection.py RFL # for RFL system
python feature_selection.py FSO # for FSO system
```

2. Then open `code_final.ipynb` to:
   - Load processed features
   - Train models
   - Generate results and visualizations

## Requirements
- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- jupyter notebook

## Data
The processed data should be placed in the `data/` directory.
# Part-B
