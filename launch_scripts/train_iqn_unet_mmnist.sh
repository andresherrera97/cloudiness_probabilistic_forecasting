#!/bin/bash
#SBATCH --job-name=iqn_mmnist
#SBATCH --ntasks=1
#SABTCH --mem=1024
#SBATCH --time=3-00:00:00
#SBATCH --partition=normal
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=97.andres.herrera@gmail.com
#SBATCH --output=sbatch_output/iqn_mmnist.out

source /etc/profile.d/modules.sh

cd /clusteruy/home/andres.herrera/cloudiness_probabilistic_forecasting
python src/train.py \
    --dataset=mmnist \
    --model_name=iqn \
    --cos_dim=64 \
    --num_bins=10 \
    --predict_diff=False \
    --epochs=30 \
    --batch_size=32 \
    --num_filters=16 \
    --save_experiment=True \
    --optimizer=adam \
    --scheduler=plateau

