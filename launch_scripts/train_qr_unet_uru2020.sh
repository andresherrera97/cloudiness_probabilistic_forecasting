#!/bin/bash
#SBATCH --job-name=qr_uru2020
#SBATCH --ntasks=1
#SBATCH --mem=1024
#SBATCH --time=72:00:00
#SBATCH --partition=normal
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=97.andres.herrera@gmail.com
#SBATCH --output=sbatch_output/qr_uru2020.out

source /etc/profile.d/modules.sh

cd /clusteruy/home/andres.herrera/cloudiness_probabilistic_forecasting
python src/train_satellite.py \
    --model_name=qr \
    --time_horizon=6 \
    --epochs=100 \
    --batch_size=8 \
    --quantiles=[0.1,0.25,0.5,0.75,0.9] \
    --num_filters=32 \
    --save_experiment=True \
    --optimizer=adam \
    --checkpoint_folder=adam \
    --scheduler=plateau \
    --train_metric=crps \
    --val_metric=crps
