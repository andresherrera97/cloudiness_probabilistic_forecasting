#!/bin/bash
#SBATCH --job-name=mean_mmnist
#SBATCH --ntasks=1
#SABTCH --mem=1024
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=97.andres.herrera@gmail.com
#SBATCH --output=sbatch_output/mean_mmnist.out

source /etc/profile.d/modules.sh

cd /clusteruy/home/andres.herrera/cloudiness_probabilistic_forecasting
python src/moving_mnist.py \
    --model_name=meanstd \
    --epochs=100 \
    --batch_size=32 \
    --num_filters=16 \
    --save_experiment=True \
    --optimizer=sgd \
    --checkpoint_folder=sgd

