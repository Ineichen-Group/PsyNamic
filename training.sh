#!/bin/bash

#SBATCH --time=0-00:30:00   ## days-hours:minutes:seconds
#SBATCH --gpus=1
#SBATCH --mem=4GB          
#SBATCH --ntasks=1          
#SBATCH --cpus-per-task=1   ## Use greater than 1 for parallelized jobs
#SBATCH --job-name=TrainPsynamic ## job name
#SBATCH --output=log/psynamic_substances.out  ## standard out file
#SBATCH --partition=lowprio

module load anaconda3
source activate psynamic_gpu_env
python model/model.py --model scibert --data data/prepared_data/substances --task 'Substances' --early_stopping_patience 3 