# Should be callablae from the command line, run_predictions.py -d data_file.csv --relevant_det True

import argparse
import pandas as pd
import os
import json
import time
import subprocess
from ast import literal_eval

PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = os.path.join(PATH, 'model_paths.json')
RUN_SCRIPTS_DIR = os.path.join(PATH, 'run_scripts')
LOG_DIR = os.path.join(PATH, 'logs')
MODEL_SCRIPT = "/home/vebern/scratch/PsyNamic/model/model.py"


def create_sbatch_file(task: str, data_file: str, model_path: str, threshold: float, pred_dir: str):
    task_name_caca= task.replace(' ', '_').replace('_', '')
    task_name_lo = task.lower().replace(' ', '_')
    data_name = os.path.splitext(data_file)[0]
    model_name = os.path.basename(os.path.dirname(model_path))

    outfile = os.path.join(pred_dir, model_name)
    job_name = f'Predict{task_name_caca}'
    log = f'{LOG_DIR}/{task_name_lo}_pred_{data_name}.out'
    sbatch_file = f'{RUN_SCRIPTS_DIR}/{task_name_lo}_{model_name}_pred_{data_name}.sh'
    # produce the bash script
    with open(sbatch_file, 'w', encoding='utf-8') as f:
        f.write(f'''#!/bin/bash
        

#SBATCH --time=0-00:05:00   ## days-hours:minutes:seconds
#SBATCH --gpus=1
#SBATCH --mem=4GB          
#SBATCH --ntasks=1          
#SBATCH --cpus-per-task=1   ## Use greater than 1 for parallelized jobs
#SBATCH --job-name={job_name} ## job name
#SBATCH --output={log}  ## standard out file
#SBATCH --partition=lowprio

python {MODEL_SCRIPT} --mode pred --load {model_path} --data {data_file} --threshold {threshold} --outfile {outfile}''')
    # make the script executable
    os.system(f'chmod +x {sbatch_file}')
    return sbatch_file


def submit_all_jobs():
    for file in os.listdir(RUN_SCRIPTS_DIR):
        if file.endswith('.sh'):
            script_path = os.path.join(RUN_SCRIPTS_DIR, file)
            process = subprocess.Popen(['sbatch', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Print output line by line
            for line in process.stdout:
                print(line, end='')
            for line in process.stderr:
                print(line, end='')

            process.wait()
    return


def check_if_jobs_done(nr_expected_outfiles: int, pred_dir: str, timeout: int = 600):
    start_time = time.time()
    while time.time() - start_time < timeout:
        outfiles = [f for f in os.listdir(pred_dir) if f.endswith('.csv')]
        if len(outfiles) == nr_expected_outfiles:
            return True
        time.sleep(1)
    raise TimeoutError(
        f"Timeout reached: Expected {nr_expected_outfiles} output files, found {len(outfiles)}")


def combine_predictions(time: str, pred_dir: str, date: str, tasks: list[dict]):
    all_preds = []
    for file in os.listdir(pred_dir):
        file_comp = file.split('_')
        task = ''
        for c in file_comp[1:]:
            # if c consists of numbers only --> date
            if c.isdigit():
                task = task.rstrip()
                break
            task += c + ' '
        
        task_dict = tasks[task.lower()]
        predictions = pd.read_csv(os.path.join(pred_dir, file))
        for i, row in predictions.iterrows():
            id2label = {int(k): v for k, v in task_dict['id2label'].items()}
            if task_dict['is_multilabel']:

                onehot_pred = literal_eval(row['prediction'])
                probabilities = literal_eval(row['probability']) 
                for i, label in enumerate(onehot_pred):
                    if label == 1:
                        output_line = {
                            'id': row['id'],
                            'task': task_dict['task'],
                            'label': id2label[i],
                            'probability': probabilities[i],
                            'model': os.path.basename(os.path.dirname(task_dict['model_path'])),
                            'is_multilabel': True
                        }
                        all_preds.append(output_line)
            else:
                prediction = row['prediction']
                probabilities = literal_eval(row['probability'])
                output_line = {
                    'id': row['id'],
                    'task': task_dict['task'],
                    'label': id2label[prediction],
                    'probability': probabilities[prediction],
                    'model': os.path.basename(os.path.dirname(task_dict['model_path'])),
                    'is_multilabel': False
                }
                all_preds.append(output_line)
        
    all_preds = pd.DataFrame(all_preds)
    all_preds.to_csv(
        f'{pred_dir}/all_predictions_{date}_{time}.csv', index=False)


def main():
    parser = argparse.ArgumentParser(
        description='Run predictions on data file')
    parser.add_argument(
        '-d', '--data', help='Data file to run predictions on', required=True)
    parser.add_argument(
        '--relevant_det', help='Whether to run relevant detections', action='store_true')
    args = parser.parse_args()

    today = pd.Timestamp.today().strftime('%Y%m%d')
    # start a timer
    start_time = time.time()

    # read in tasks from model_paths:
    with open(MODEL_PATHS, 'r', encoding='utf-8') as f:
        tasks = json.load(f)

    # create prediction directory with today's date
    prediction_dir = os.path.join(PATH, 'predictions', today)
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir, exist_ok=True)

    data = args.data
    if args.relevant_det:
        pred_outfile = os.path.join(
            prediction_dir, f'{args.data}_relevant_predictions.csv')
        for task_dict in tasks:
            if task_dict['task'] == "Relevant":
                create_sbatch_file(
                    task_dict['task'], args.data, task_dict['model_path'], task_dict['prediction_threshold'], prediction_dir)

        submit_all_jobs()
        check_if_jobs_done(1, prediction_dir, timeout=1800)
        # TODO: filter only relevant detections
        data = pred_outfile

        # remove all scripts in the run_scripts directory
        for file in os.listdir(RUN_SCRIPTS_DIR):
            os.remove(os.path.join(RUN_SCRIPTS_DIR, file))

    nr_jobs = len(tasks) if args.relevant_det else len(tasks) - 1
    for task_dict in tasks:
        if task_dict['task'] == "Relevant":
            continue
        create_sbatch_file(
            task_dict['task'], data, task_dict['model_path'], task_dict['prediction_threshold'], prediction_dir)

    submit_all_jobs()
    check_if_jobs_done(nr_jobs, prediction_dir, timeout=1800)
    time_elapsed = time.time() - start_time
    time_elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time_elapsed))

    tasks = {task['task'].lower(): task for task in tasks}

    combine_predictions(time_elapsed_str, prediction_dir, today, tasks)


if __name__ == "__main__":
    main()
