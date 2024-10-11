#!/bin/bash
#SBATCH --job-name=det_salto
#SBATCH --ntasks=1
#SBATCH --mem=1024
#SBATCH --time=5-00:00:00
#SBATCH --partition=normal
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=97.andres.herrera@gmail.com
#SBATCH --output=sbatch_output/det_salto.out

source /etc/profile.d/modules.sh

cd /clusteruy/home/andres.herrera/cloudiness_probabilistic_forecasting
python src/train.py \
    --dataset=goes16 \
    --model_name=det \
    --epochs=20 \
    --batch_size=6 \
    --num_filters=16 \
    --save_experiment=True \
    --optimizer=adam \
    --scheduler=plateau
