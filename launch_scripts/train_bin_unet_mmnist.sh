#!/bin/bash
#SBATCH --job-name=bin_mmnist
#SBATCH --ntasks=1
#SABTCH --mem=1024
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=97.andres.herrera@gmail.com
#SBATCH --output=sbatch_output/bin_mmnist.out

source /etc/profile.d/modules.sh

cd /clusteruy/home/andres.herrera/cloudiness_probabilistic_forecasting
python src/moving_mnist.py \
    --model_name=bin_classifier \
    --epochs=100 \
    --batch_size=32 \
    --num_bins=5 \
    --num_filters=32 \
    --save_experiment=True \
    --optimizer=adam \
    --train_metric=crps \
    --val_metric=crps \
    --checkpoint_folder=five_bins_32_filters \
    --scheduler=plateau

