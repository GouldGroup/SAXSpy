# SAXSpy
This repository contains the synthetic data generation, processing and model training code for predicting lipid phase behaviour, details are in the paper titled <em>
"Machine Learning Platform for Determining Experimental Lipid Phase Behaviour from Small Angle X-ray Scattering Patterns by Pre-training on Synthetic Data".</em>

## Getting Started

### Requirements

To get started, install the requirements by:

`pip install -r requirements.txt`

### Quick Start

To begin, you need to generate some synthetic SAXS Samples. Let's generate some lamellar phase samples to begin, navigate to `Generation_scripts` and run
```
python Generate_Lamellar.py
```
This should automatically generate a set of lamellar I/q values and save it into `Synthetic_raw`. A random sample from the generated data will also be plotted.

Note: The default parameters for synthetic data generation are the ones that gave a resulting matching distribution with real, experimental SAXS patterns - feel free to experiment with different ranges!

## Data Generation
There are 3 synthetic data generation scripts in `Generation_scripts`. Each script uses our saxspy module to allow you to generate entire datasets of SAXS patterns for a particular lipid phase. The cubic phase script, `Generate_Cubic.py` can be modified to generate patterns for Primitive, Gyroid or Diamond surface cubic phases. This can be done by varying the phase variable 

```python
# Instantiate the synthetic model: 'P', 'G', or 'D'
phase = 'G'
```
Each scripts generates the data based on the params variable - a list of parameter ranges.

```python
# ranges of: lattice parameter, head position, sigma head, sigmal tail
params = np.array([[20, 78], [5, 30], [0.5, 3], [0.5, 5]])
```
The resulting datasets are saved as I/q arrays in the `Synthetic_raw` folder.

### Verbosity
Each generation script has a `verbose` boolean, when `True`, a random sample from the generated data is plotted. 

## Data Processing
Once the synthetic data is generated, we perform multiple pre-processing steps before massing through the model. This can be done with the scripts in `Preprocessing_scripts`. In order to correctly pre-process the data for training, ensure that your raw data files are in the `Synthetic_raw` directory with the correct phase names as given by the generation scripts. The processed datafiles will be saved in 

## Model Training
`train.py` is the training/validation script. Assuming you have generated and processed all available phases, running `python train.py` from this repository's root should start training your model on the synthetic data. The trained model will be saved as `trained_saxs_model.h5`.