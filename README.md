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
1. Download the example data from the following [link:](https://drive.google.com/drive/folders/1ee7xGS2pCp-vJqfjuuMSpEZ_EmV_g6Pa?usp=drive_link)
2. Uncompress the data.
3. Modify the `data_root` variable in `configs/paths/paths_panda.yaml`. It should point to the `PandaSet` directory. For example: `"/media/norbert/T7/PandaSet"`
4. Download the generated ground truth from the following [link:](https://drive.google.com/drive/folders/1ee7xGS2pCp-vJqfjuuMSpEZ_EmV_g6Pa?usp=drive_link) (`gt_panda` folder)
5. Put the ground truth folder into the `results` directory.

## Run Code

> 💡 Before running any of the scripts, go into the repository root, and activate the conda environment as described above.


To run the algorithm described in the paper, use the main script:
```bash
python src/run_exp_hg_panda_ts.py
```

You can also run the SOTA method, using the following python script:
```bash
python src/run_ref_gnf_panda.py
```

Run the evaluation script, after you ran both our method and the SOTA method for a certain sequence (default: 039):
```bash
python src/eval/eval.py
```


You can re-generate the ground truth (for any sequence) using the following script (set the `data_root` and `seq_num` in `configs/paths/paths_panda.yaml`):
```bash
python src/gt/gen_gt_normal_panda.py
```
