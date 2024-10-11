#!/bin/bash
#SBATCH --job-name=sep-2022
#SBATCH --ntasks=1
#SBATCH --mem=32768
#SBATCH --time=1-00:00:00
#SBATCH --partition=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=97.andres.herrera@gmail.com
#SBATCH --output=sbatch_output/september-2022.out

source /etc/profile.d/modules.sh

cd /clusteruy/home/andres.herrera/cloudiness_probabilistic_forecasting
python src/goes16_dataset_generator.py \
    --start_date=2022-09-13 \
    --end_date=2022-09-13 \
    --lat=-31.2827 \
    --lon=-57.9181 \
    --size=1024 \
    --output_folder=datasets/goes16/salto/ \
    --skip_night=True \
    --save_only_first=False


