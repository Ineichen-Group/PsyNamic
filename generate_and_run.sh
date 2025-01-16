#!/bin/bash

# Model name
model="pubmedbert"

# List of tasks
tasks=(
    "Data Collection"
    "Data Type"
    "Number of Participants"
    "Age of Participants"
    "Application Form"
    "Clinical Trial Phase"
    "Condition"
    "Outcomes"
    "Regimen"
    "Setting"
    "Study Control"
    "Study Purpose"
    "Substance Naivety"
    "Substances"
    "Sex of Participants"
    "Study Conclusion"
    "Study Type"
    "NER Bio"
)

# Directory to store the generated scripts
mkdir -p training_scripts
mkdir -p log

# Iterate over each task
for task in "${tasks[@]}"; do
    # Convert the task name to a filename-friendly format
    task_filename=$(echo "$task" | tr ' ' '_' | tr '[:upper:]' '[:lower:]')

    # Create the training script for the task
    cat <<EOT > training_scripts/train_${model}_${task_filename}.sh
#!/bin/bash

#SBATCH --time=0-00:15:00   ## days-hours:minutes:seconds
#SBATCH --gpus=1
#SBATCH --mem=4GB          
#SBATCH --ntasks=1          
#SBATCH --cpus-per-task=1   ## Use greater than 1 for parallelized jobs
#SBATCH --job-name=TrainPsynamic_${model}_${task_filename} ## job name
#SBATCH --output=log/psynamic_${model}_${task_filename}.out  ## standard out file
#SBATCH --partition=lowprio

module load anaconda3
source activate psynamic_gpu_env
python model/model.py --model ${model} --data data/prepared_data/${task_filename} --task '${task}' --early_stopping_patience 3 
EOT

    # Make the generated script executable
    chmod +x training_scripts/train_${model}_${task_filename}.sh

    # Submit the job
    sbatch training_scripts/train_${model}_${task_filename}.sh

done
