#!/bin/bash
#SBATCH --job-name=mcd_mmnist
#SBATCH --ntasks=1
#SABTCH --mem=1024
#SBATCH --time=3-00:00:00
#SBATCH --partition=normal
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=97.andres.herrera@gmail.com
#SBATCH --output=sbatch_output/mcd_mmnist.out

source /etc/profile.d/modules.sh

cd /clusteruy/home/andres.herrera/cloudiness_probabilistic_forecasting
python src/moving_mnist.py --model_name=mcd --epochs=20 --batch_size=32 --dropout_p=0.5 --num_ensemble_preds=5 --num_filters=16 --save_experiment=True --optimizer=sgd

