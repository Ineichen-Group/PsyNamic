#!/bin/bash

#SBATCH --time=0-00:10:00   ## days-hours:minutes:seconds
#SBATCH --gpus=1
#SBATCH --mem=4GB          
#SBATCH --ntasks=1          
#SBATCH --cpus-per-task=1   ## Use greater than 1 for parallelized jobs
#SBATCH --job-name=PredictPsynamic ## job name
#SBATCH --output=log/psynamic_eval_substances.out  ## standard out file
#SBATCH --partition=lowprio

python model/model.py --mode eval --load model/experiments/clinicalbert_sex_of_participants_20250221/checkpoint-480 --outfile test