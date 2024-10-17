import argparse
import json
import os
from ast import literal_eval
from datetime import datetime
from typing import Union, Literal

import numpy as np
import pandas as pd
import torch
from confidenceinterval import classification_report_with_ci
from datahandler import (MODEL_IDENTIFIER, DataHandler, DataHandlerBIO,
                         DataSplit, DataSplitBIO, SimpleDataset)
from evaluate import load
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch.optim import AdamW
from transformers import (AutoModelForSequenceClassification,
                          AutoModelForTokenClassification, AutoTokenizer,
                          BertForSequenceClassification,
                          BertForTokenClassification,
                          EarlyStoppingCallback, Trainer, TrainingArguments,
                          get_linear_schedule_with_warmup)
from transformers.trainer_utils import PredictionOutput

############################################################################################################
# REFERENCES
# rf. https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb#scrollTo=797b2WHJqUgZ for multilabel tuning
############################################################################################################
# TODO: Enable cross validation
# TODO: Hyperparameter tune
# TODO: Improve predicting whatever file (single label or multilabel)


############################################################################################################
# some globals to set
EXPERIMENT_PATH = './model/experiments'
os.environ['WANDB_PROJECT'] = "psynamic" # Used for logging to wandb
############################################################################################################


############################################################################################################
# METRICS
############################################################################################################
def singlelabel_metrics(labels: list[int], preds: list[int]) -> dict[str, float]:
    """Compute metrics for binary classification tasks."""
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(
        labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def multilabel_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    """Compute metrics for multilabel classification tasks."""
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(
        labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def bio_metrics(labels: list[str], preds: list[str]) -> dict[str, float]:
    metric = load("seqeval")
    results = metric.compute(
        predictions=preds, references=labels)
    final_results = {}
    final_results["overall_precision"] = results["overall_precision"]
    final_results["overall_recall"] = results["overall_recall"]
    final_results["overall_f1"] = results["overall_f1"]
    final_results["overall_accuracy"] = results["overall_accuracy"]

    for key, value in results.items():
        if isinstance(value, dict):
            for n, v in value.items():
                final_results[f"{key}_{n}"] = v

    Ps, Rs, Fs = [], [], []
    for type_name in results:
        if type_name.startswith("overall"):
            continue
        Ps.append(results[type_name]["precision"])
        Rs.append(results[type_name]["recall"])
        Fs.append(results[type_name]["f1"])

    final_results["macro_precision"] = np.mean(Ps)
    final_results["macro_recall"] = np.mean(Rs)
    final_results["macro_f1"] = np.mean(Fs)

    return final_results


def compute_singlelabel_metrics(pred: PredictionOutput) -> dict[str, float]:
    """Metric for binary classification, to be passed to the Trainer"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return singlelabel_metrics(labels, preds)


def compute_multilabel_metrics(pred: PredictionOutput, threshold: float = 0.1, average: Literal['micro', 'macro', 'samples', 'weighted'] = 'micro') -> dict[str, float]:
    """Compute metrics for multilabel classification tasks, to be passed to the Trainer.

    Args:
        pred (PredictionOutput): PredictionOutput object from the Trainer
        threshold (float, optional): Threshold for prediction probabilities. Defaults to 0.1. VERY IMPORTANT!
        average (Literal['micro', 'macro', 'samples', 'weighted'], optional): 
            Average type for metrics. Defaults to 'micro'.
            # s. https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.f1_score.html for more information
    """

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(pred.predictions))
    labels = pred.label_ids
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1

    return multilabel_metrics(labels, y_pred)


def compute_bio_metrics(p: PredictionOutput, label_list: list[str]) -> dict[str, float]:
    """Compute metrics for NER tasks, to be passed to the Trainer.
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return bio_metrics(true_labels, true_predictions)

############################################################################################################
# COMMAND LINE INTERFACE
############################################################################################################
def init_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str,
                        choices=['train', 'cont_train', 'eval', 'pred'], default='train')
    parser.add_argument('--data', type=str)
    # If NER, the task name should include 'NER'
    parser.add_argument('--task', type=str)

    # TRAIN:
    # General
    parser.add_argument('--model', type=str)
    # TODO: Implement cross validation
    parser.add_argument('--cross_val', type=bool, default=False)

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lr_scheduler', type=str, default='linear')
    parser.add_argument('--warmup_ratio', type=int, default=0.1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--early_stopping_patience', type=int, default=3)
    parser.add_argument('--gradient_clipping', type=float, default=0.1)
    parser.add_argument('--device', type=str,
                        choices=['cpu', 'cuda'], default='cuda')

    # CONT_TRAIN
    parser.add_argument('--load', type=str, default=None)  # Experiment folder

    return parser.parse_args()


############################################################################################################
# ALL SORTS OF HELPER FUNCTIONS
############################################################################################################
def save_train_args(project_dir: str, args: argparse.Namespace, add_params: dict = None) -> None:
    """ Helper function to save training arguments to a json file."""
    args = vars(args)
    if add_params is not None:
        args.update(add_params)
    with open(os.path.join(project_dir, 'params.json'), 'w') as f:
        json.dump(args, f)


def set_args_from_file(args: argparse.Namespace) -> argparse.Namespace:
    """Set arguments from a params.json file in the experiment folder."""
    params_json = os.path.join(os.path.dirname(args.load), 'params.json')
    with open(params_json, 'r') as f:
        params = json.load(f)

    ignore = ['mode', 'load']
    if args.data is not None:
        ignore.append('data')

    for key, value in params.items():
        if key not in ignore:
            setattr(args, key, value)

    return args


def init_directories(model: str, task: str) -> str:
    """ Helper function to initialize directory for the experiment. --> make sure to set EXPERIMENT_PATH
        The directory will be names: {model}_{task}_{date}
    """
    date = datetime.now().strftime("%Y%m%d")
    lower_case_task = task.lower().replace(' ', '_')
    experiment_path = f'{model}_{lower_case_task}_{date}'
    project_dir = os.path.join(EXPERIMENT_PATH, experiment_path)
    os.makedirs(project_dir, exist_ok=True)
    return project_dir


def load_model(args: argparse.Namespace) -> Trainer:
    """Load a model from a given path and return a Trainer object."""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if 'NER' in args.task:
        model = AutoModelForTokenClassification.from_pretrained(
            args.load).to(device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.load).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.load)
    trainer = Trainer(
        model=model,
        compute_metrics=compute_multilabel_metrics,
        tokenizer=tokenizer)
    return trainer


def load_data(data_dir: str, meta_file: str, model_identifier: str) -> tuple[DataSplit, DataSplit, DataSplit]:
    """ Helper function to load data splits which were previously created by the DataHandler.
        The data_dir is expected to contain the following:
        - meta.json
        - test.csv
        - train.csv
        (- val.csv) if validation split was used
        all csv are expected to have an 'id', 'text' and 'label' column.

    """
    # open meta file
    meta_data = json.load(open(meta_file, 'r'))
    if 'NER' in meta_data['Task']:
        datahandler = DataHandlerBIO(data_dir, model=model_identifier)
        use_val = datahandler.load_splits(data_dir)

        train_dataset, test_dataset, eval_dataset = datahandler.get_split()

    else:
        datahandler = DataHandler(
            data_path=data_dir, model=model_identifier, meta_file=meta_file)
        # TODO: clean up usage of use_val
        use_val = datahandler.load_splits(data_dir)
        train_dataset, test_dataset, eval_dataset = datahandler.get_strat_split(
            use_val=use_val)

    return train_dataset, test_dataset, eval_dataset


def train(
    project_dir: str,
    train_dataset: Union[DataSplit, DataSplitBIO],
    args: argparse.Namespace,
    val_dataset: Union[DataSplit, DataSplitBIO] = None,
    resume_from_checkpoint: bool = False,
    is_multilabel: bool = False,
) -> Trainer:
    device_name = args.device if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    problem_type = "single_label_classification" if not is_multilabel else "multi_label_classification"

    # Load checkpoint or initialize model
    if resume_from_checkpoint:
        tokenizer = AutoTokenizer.from_pretrained(args.load)
        if 'NER' in args.task:
            model = AutoModelForTokenClassification.from_pretrained(
                args.load).to(device)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                args.load, problem_type=problem_type).to(device)
    else:
        model_id = MODEL_IDENTIFIER[args.model]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if 'NER' in args.task:
            model = BertForTokenClassification.from_pretrained(
                model_id, num_labels=train_dataset.nr_labels).to(device)
        else:
            model = BertForSequenceClassification.from_pretrained(
                model_id, num_labels=train_dataset.nr_labels, problem_type=problem_type).to(device)

    training_args = TrainingArguments(
        output_dir=project_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch" if val_dataset is not None else "no",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to='wandb',
        logging_dir=project_dir,
        logging_strategy="epoch",
        run_name=os.path.basename(project_dir),
        resume_from_checkpoint=args.load if resume_from_checkpoint else None,
        metric_for_best_model='eval_loss' if val_dataset is not None else None,
        use_cpu=device_name == 'cpu',
    )

    total_steps = len(train_dataset) * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.learning_rate,
                      weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps
    )

    # Chose metrics based on task
    if 'NER' in args.task:
        label_list = train_dataset.labels
        def metrics(p): return compute_bio_metrics(
            (p.predictions, p.label_ids), label_list)
    elif is_multilabel:
        metrics = compute_multilabel_metrics
    else:
        metrics = compute_singlelabel_metrics

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=metrics,
        optimizers=(optimizer, scheduler),
    )

    if args.early_stopping_patience > 0 and val_dataset is not None:
        trainer.add_callback(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience))

    # Train the model
    trainer.train()

    return trainer


def predict(project_folder: str, trainer: Trainer, test_dataset: Union[DataSplit, DataSplitBIO], outfile: str = None) -> str:
    """ Predicts the labels for the test dataset and saves the predictions to a CSV file.

    Args:
        project_folder (str): Path in which the checkpoints, params.json and predictions.csv will be saved.
        trainer (Trainer): Trainer object with the trained model.
        test_dataset (DataSplit): Test dataset to predict on.
        outfile (str, optional): Name of the output file. Defaults to predictions.csv in project_folder.

    Returns:
        str: outfile path
    """
    # Get predictions
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions

    # Calculate probabilities using softmax
    probabilities = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    
    is_ner = type(test_dataset) == DataSplitBIO
    
    # Prepare lists for DataFrame
    data = []
    for i, (id, text, label) in enumerate(test_dataset):
        if test_dataset.is_multilabel:
            # For multilabel classification
            # Convert probabilities to binary predictions
            prediction = (probabilities[i] > 0.5).astype(int).tolist()
            # Save all probabilities for each label
            probability = probabilities[i].tolist()
            # Ensure label is also in binary format
            label = literal_eval(label)
        else:
            # For single-label classification
            prediction = np.argmax(logits[i])
            probability = probabilities[i].tolist()
            label = label

        data.append({
            "id": id,
            "text": text,
            "prediction": prediction,
            "probability": probability,
            "label": label
        })

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save DataFrame to CSV
    filename = 'predictions.csv' if outfile is None else f'{outfile}_predictions.csv'
    output_file = os.path.join(project_folder, filename)
    df.to_csv(output_file, index=False)
    return output_file


def evaluate(project_folder: str, trainer: Trainer, test_dataset: Union[DataSplit, DataSplitBIO], task: str) -> str:
    output_file = predict(project_folder, trainer, test_dataset)
    # Read in predictions
    df = pd.read_csv(output_file)
    # TODO: really hacky this whole literal_eval, should be dealt with nicer
    if 'NER' not in task:
        # convert string to list
        df['label'] = df['label'].apply(literal_eval)
        df['prediction'] = df['prediction'].apply(literal_eval)
        
    predictions = df['prediction'].tolist()
    labels = df['label'].tolist()
    if 'NER' in task:
        predictions = [literal_eval(list_string) for list_string in predictions]
        labels = [literal_eval(list_string) for list_string in labels]
    report_df = None
    # Compute classification report and save to CSV
    if test_dataset.is_multilabel:
        metrics = multilabel_metrics(labels, predictions)

    else:
        if 'NER' in task:
            metrics = bio_metrics(labels, predictions)
        else:
            try:
                report_df = classification_report_with_ci(labels, predictions,
                                                          output_dict=True)
            except Exception as e:
                print('Error computing classification report for single-label task:', e)
        metrics = singlelabel_metrics(labels, predictions)

    output_file = open(os.path.join(project_folder, 'metrics.json'), 'w')
    # write metrics and/or classification report to file
    with output_file as f:
        json.dump(metrics, f)

        if report_df is not None:
            # append classification report to metrics.json
            json.dump(report_df, f)
    return output_file


############################################################################################################
# 4 MODES: train, cont_train, eval, pred
############################################################################################################
def finetune(args: argparse.Namespace) -> None:
    """ Finetune a pretrained BERT model on a given dataset, where the splits were created by the DataHandler.
        Example call for MODE='train' e.g. python model/model.py --model pubmedbert --data data/prepared_data/asreview_dataset_all --task 'Relevant Sample'
    """
    project_path = init_directories(args.model, args.task)
    meta_file = os.path.join(args.data, 'meta.json')
    meta_data = json.load(open(meta_file, 'r'))
    # TODO: solve it nice that meta data is not loaded twice
    train_dataset, test_dataset, val_dataset = load_data(
        args.data, meta_file, args.model)
    add_params = {
        'max_length': train_dataset.max_len,
        'is_multilabel': meta_data['Is_multilabel'],
    }
    save_train_args(project_path, args, add_params)
    trainer = train(project_path, train_dataset,
                    args, val_dataset=val_dataset, is_multilabel=meta_data['Is_multilabel'])
    evaluate(project_path, trainer, test_dataset, meta_data['Task'])


def cont_finetune(args: argparse.Namespace) -> None:
    """ Continue training a model from a given path.
        Example call for MODE='cont_train', e.g. python model/model.py --mode cont_train --load model/experiments/pubmedbert_relevant_sample_20240730/checkpoint-565
    """
    args = set_args_from_file(args)
    meta_file = os.path.join(args.data, 'meta.json')
    meta_data = json.load(open(meta_file, 'r'))
    train_dataset, test_dataset, eval_dataloader = load_data(
        args.data, meta_file, args.model)
    trainer = train(args.load, train_dataset,
                    args, resume_from_checkpoint=True, is_multilabel=meta_data['Is_multilabel'], task=meta_data['Task'])
    evaluate(args.load, trainer, test_dataset, meta_data['Task'])


def load_and_evaluate(args: argparse.Namespace) -> str:
    """ Load a model from a given path and evaluate it on a test dataset. 
        Example call for MODE='eval', e.g. python model/model.py --mode eval --load model/experiments/pubmedbert_relevant_sample_20240730/checkpoint-565
    """
    # Load model from path
    args = set_args_from_file(args)
    trainer = load_model(args)

    exp_path = os.path.dirname(args.load)
    data_meta_file = os.path.join(args.data, 'meta.json')

    test_dataset = load_data(
        args.data, data_meta_file, args.model)[1]
    evaluate(exp_path, trainer, test_dataset, args.task)


def load_and_predict(args: argparse.Namespace) -> None:
    """ Load a model from a given path and predict on a given dataset, either a test DataSplit or a file

        If args.data is not specified, the test split in the data directory used for training will be used.
        If args.data is a file, the model will predict on this file. The file is expected to have an 'id' and 'text' column.

        Example call for MODE = 'pred': python model/model.py --mode pred --load model/experiments/pubmedbert_relevant_sample_20240730/checkpoint-565 --data data/prepared_data/all/psychedelic_study_relevant.csv
    """
    # Load model from path
    args = set_args_from_file(args)
    trainer = load_model(args)
    exp_path = os.path.dirname(args.load)
    data_meta_file = os.path.join(args.data, 'meta.json')

    # Case 1: Test split of training used
    if args.data is None:
        dataset = load_data(
            args.data, data_meta_file, args.model)[1]
        outfile_name = f'{os.path.basename(args.load)}_test_split'
        outfile = predict(exp_path, trainer, dataset, outfile_name)
    else:
        if os.path.isfile(args.data):
            data = args.data
        else:
            raise ValueError('Please provide a valid path to the data file')
        model = MODEL_IDENTIFIER[args.model]
        tokenizer = AutoTokenizer.from_pretrained(model)
        max_length = args.max_length
        dataset = SimpleDataset(
            data, tokenizer, max_length, args.is_multilabel)
        outfile_name = f'{os.path.basename(args.load)}_{os.path.basename(data).split(".")[0]}'
        outfile = predict(exp_path, trainer, dataset, outfile_name)
    return outfile


def main():
    args = init_argparse()
    if args.mode == 'train':
        finetune(args)

    elif args.mode == 'cont_train':
        # check if the experiment folder exists
        if args.load is None or not os.path.exists(args.load):
            raise ValueError(
                'Please provide the correct path to the experiment folder to continue training')

        cont_finetune(args)

    elif args.mode == 'eval':
        if args.load is None or not os.path.exists(args.load):
            raise ValueError(
                'Please provide the correct path to the experiment folder to continue training')

        if args.data is None:
            print('Defaulting to test data split used for training')

        load_and_evaluate(args)
    elif args.mode == 'pred':
        if args.load is None or not os.path.exists(args.load):
            raise ValueError(
                'Please provide the correct path to the experiment folder to continue training')

        load_and_predict(args)


if __name__ == "__main__":
    main()
