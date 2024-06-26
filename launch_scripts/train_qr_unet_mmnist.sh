#!/bin/bash
#SBATCH --job-name=qr_mmnist
#SBATCH --ntasks=1
#SABTCH --mem=1024
#SBATCH --time=72:00:00
#SBATCH --partition=normal
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=97.andres.herrera@gmail.com
#SBATCH --output=sbatch_output/qr_mmnist.out

source /etc/profile.d/modules.sh

cd /clusteruy/home/andres.herrera/cloudiness_probabilistic_forecasting
python src/moving_mnist.py \
    --model_name=qr \
    --epochs=100 \
    --batch_size=32 \
    --quantiles=[0.05,0.1,0.25,0.4,0.5,0.60.75,0.9,0.95] \
    --num_filters=16 \
    --save_experiment=True \
    --optimizer=adam \
    --checkpoint_folder=adam \
    --scheduler=plateau \
    --checkpoint_metric=crps

