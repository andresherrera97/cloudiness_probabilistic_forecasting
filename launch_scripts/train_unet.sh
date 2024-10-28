#!/bin/bash
#SBATCH --job-name=unet_time_horizon
#SBATCH --ntasks=1
#SBATCH --mem=8192
#SBATCH --time=3-00:00:00
#SBATCH --partition=normal
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=97.andres.herrera@gmail.com
#SBATCH --output=sbatch_output/unet_time_horizon.out

source /etc/profile.d/modules.sh

cd /clusteruy/home/andres.herrera/cloudiness_probabilistic_forecasting
python src/train.py \
    --dataset=goes16 \
    --model_name=unet_type \
    --checkpoint_folder=None \
    --spatial_context=0 \
    --num_bins=None \
    --epochs=50 \
    --batch_size=4 \
    --num_filters=32 \
    --save_experiment=True \
    --optimizer=adam \
    --scheduler=plateau \
    --input_frames=3
