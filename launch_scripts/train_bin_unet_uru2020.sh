#!/bin/bash
#SBATCH --job-name=bin_uru2020
#SBATCH --ntasks=1
#SABTCH --mem=1024
#SBATCH --time=72:00:00
#SBATCH --partition=normal
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=97.andres.herrera@gmail.com
#SBATCH --output=sbatch_output/bin_uru2020.out

source /etc/profile.d/modules.sh

cd /clusteruy/home/andres.herrera/cloudiness_probabilistic_forecasting
python src/train_satellite.py \
    --model_name=bin_classifier \
    --time_horizon=6 \
    --epochs=100 \
    --batch_size=8 \
    --num_bins=5 \
    --num_filters=16 \
    --save_experiment=True \
    --optimizer=adam \
    --train_metric=crps \
    --val_metric=crps \
    --checkpoint_folder=five_bins_16_filters_uru \
    --scheduler=plateau
