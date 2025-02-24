# ground-normal-prediction
Code and supplementary material for the paper.

## Clone the Repository

```bash
# clone project (with submodules)
git clone --recurse-submodules https://github.com/norbertmarko/ground-normal-prediction
```
```bash
git submodule update --init --recursive
```

## Installation

Create environment (you need [miniconda](https://docs.anaconda.com/miniconda/install/) for this).

```bash
# create conda environment and install dependencies
conda env create -f environment.yaml

# activate conda environment (every new console used)
conda activate ground-normal-prediction
```

Install `pandaset-devkit` in the environment.

```bash
cd src/_ref/pandaset_devkit
```
Activate your conda environment (this is where we pip install the devkit)
```bash
conda activate ground-normal-prediction
```

`cd` into `pandaset_devkit/python` (assuming you are already in `pandaset_devkit`)
```bash
cd python
```
Install the devkit
```bash
pip install .
```

## Prepare Evaluation

## Run Code
