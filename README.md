### Evaluation

Run evaluation in the `unitraj` conda environment configured following the [UniTraj installation instructions](https://github.com/vita-epfl/UniTraj#-quick-start).

Evaluation uses the nuScenes validation data cached in HDF5 format (`.h5`) by UniTraj. Set `H5_DIR` to the directory containing these cache files.

```bash
# Genarate val dataset
conda activate unitraj
cd path/to/scenarionet # This can be find in "Unitraj --> README.md --> Data Preparation"

python -m scenarionet.convert_nuscenes \
  --split val \
  --dataroot path/to/nuscenes \
  --database_path path/to/unitraj/cache/nuscenes_val_prediction/data \
  --num_workers 8
export H5_DIR="path/to/unitraj/cache/nuscenes_val_prediction/data"


# Compute metric
conda activate unitraj
cd path/to/mdp-code

# Evaluate Forecast-MAE.
CUDA_VISIBLE_DEVICES=2 python evaluate.py \
  method=forecast_mae \
  h5_dir="$H5_DIR"

# Evaluate MDP.
CUDA_VISIBLE_DEVICES=2 python evaluate.py \
  method=mdp \
  h5_dir="$H5_DIR"

# Compare Forecast-MAE and MDP results.
python hard_test.py
```

Replace the placeholder paths with your local paths and adjust `CUDA_VISIBLE_DEVICES` to select an available GPU.