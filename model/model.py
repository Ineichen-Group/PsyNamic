import argparse
import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from confidenceinterval import classification_report_with_ci
from datahandler import DataHandler, PsychNamicRelevant, PsyNamicSingleLabel, DataSplit
from abc import ABC, abstractmethod

# rf. https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb#scrollTo=797b2WHJqUgZ for multilabel tuning


EXPERIMENT_PATH = './model/experiments'
DATA_PATH = './data/prepared_data'
MODEL_IDENTIFIER = {
    'pubmedbert': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
    'biomedbert-abstract': 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',
    'scibert': 'allenai/scibert_scivocab_uncased',
    'biobert': 'dmis-lab/biobert-v1.1',
    'clinicalbert': 'emilyalsentzer/Bio_ClinicalBERT',
    'biolinkbert': 'michiyasunaga/BioLinkBERT-base',
}

os.environ['WANDB_PROJECT'] = "psynamic"
# TODO: Enable cross validation
# TODO: enable all hyperparameters
# TODO: Hyperparameter tune


class CustomTrainer:

    def __innit__(self, arg: argparse.Namespace):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def load_model(self):
        pass


class BertTrainer(CustomTrainer):

    pass


def init_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str,
                        choices=['train', 'cont_train', 'eval'], default='train')
    parser.add_argument('--data', type=str)
    parser.add_argument('--task', type=str)

    # TRAIN:
    # General
    parser.add_argument('--model', type=str)
    parser.add_argument('--cross_val', type=bool, default=False)

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lr_scheduler', type=str, default='linear')
    parser.add_argument('--warmup_ratio', type=int, default=0.1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--early_stopping_patience', type=int, default=3)
    parser.add_argument('--gradient_clipping', type=float, default=0.1)
    parser.add_argument(
        '--metrics', type=list[str], default=['accuracy', 'precision', 'recall', 'f1'])
    parser.add_argument('--device', type=str,
                        choices=['cpu', 'cuda'], default='cuda')

    # CONT_TRAIN
    parser.add_argument('--load', type=str, default=None)  # Experiment folder

    return parser.parse_args()


def save_train_args(project_dir: str, args: argparse.Namespace) -> None:
    # Write the training arguments to a json
    with open(os.path.join(project_dir, 'params.json'), 'w') as f:
        json.dump(vars(args), f)


def init_directories(model: str, task: str) -> str:
    date = datetime.now().strftime("%Y%m%d")
    lower_case_task = task.lower().replace(' ', '_')
    experiment_path = f'{model}_{lower_case_task}_{date}'
    project_dir = os.path.join(EXPERIMENT_PATH, experiment_path)
    os.makedirs(project_dir, exist_ok=True)
    return project_dir


def load_data(data: str, meta_file: str) -> tuple[DataSplit, DataSplit, DataSplit]:
    datahandler = DataHandler(data, meta_file=meta_file)
    use_val = datahandler.load_splits(data)
    train_dataset, test_dataset, eval_dataset = datahandler.get_strat_split(
        use_val=use_val)
    return train_dataset, test_dataset, eval_dataset


def train(
    project_dir: str,
    train_dataset,
    test_dataset,
    args,
    val_dataset=None,  # Optional parameter for validation dataset
    resume_from_checkpoint=False,
    is_multilable=False
) -> Trainer:
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    problem_type = "single_label_classification" if not is_multilable else "multi_label_classification"

    if resume_from_checkpoint:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.load,
            problem_type=problem_type).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.load)
    else:
        model_id = MODEL_IDENTIFIER[args.model]
        model = BertForSequenceClassification.from_pretrained(
            model_id, num_labels=train_dataset.nr_labels,
            problem_type=problem_type).to(device)
        tokenizer = BertTokenizer.from_pretrained(model_id)

    training_args = TrainingArguments(
        output_dir=project_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        # Conditionally set evaluation strategy
        eval_strategy="epoch" if val_dataset is not None else "no",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to='wandb',
        logging_dir=project_dir,
        logging_strategy="epoch",
        run_name=os.path.basename(project_dir),
        resume_from_checkpoint=args.load if resume_from_checkpoint else None,
        metric_for_best_model='eval_loss' if val_dataset is not None else None
    )

    total_steps = len(train_dataset) * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.learning_rate,
                      weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(training_args.warmup_ratio * total_steps),
        num_training_steps=total_steps
    )

    # Define compute metrics function
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
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
        
    
    def multi_label_metrics(pred):
        threshold = 0.5
        # Apply sigmoid to predictions
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(pred.predictions))
        labels = pred.label_ids
        # Apply threshold to convert probabilities to binary predictions
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        
        # Calculate metrics
        y_true = labels
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        roc_auc = roc_auc_score(y_true, y_pred, average='micro')
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='micro')
        recall = recall_score(y_true, y_pred, average='micro')
        
        # Return metrics as a dictionary
        metrics = {
            'f1': f1_micro_average,
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }
        
        return metrics

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # Conditionally pass validation dataset
        eval_dataset=val_dataset if val_dataset is not None else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if not is_multilable else multi_label_metrics,
        optimizers=(optimizer, scheduler),
    )

    if args.early_stopping_patience > 0 and val_dataset is not None:
        trainer.add_callback(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience))

    # Train the model
    trainer.train()

    return trainer


def evaluate(project_folder: str, trainer: Trainer, test_dataset: DataSplit) -> str:
    # Get predictions
    predictions = trainer.predict(test_dataset)
    labels = predictions.label_ids
    preds = predictions.predictions.argmax(-1)

    # Calculate probabilities using softmax
    logits = predictions.predictions
    probabilities = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)

    # Prepare lists for DataFrame
    data = []
    for i, (id, text, label) in enumerate(test_dataset):
        prediction = preds[i]
        probability = probabilities[i][prediction]
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
    output_file = os.path.join(project_folder, 'predictions.csv')
    df.to_csv(output_file, index=False)

    # Compute classification report and save to CSV
    y_true = labels
    y_predicted = preds
    # if there is only one class, the classification report cannot be computed
    if len(set(y_predicted)) == 1:
        print('Only one class present in the predictions, classification report cannot be computed.')

    else:
        try:
            report_df = classification_report_with_ci(y_true, y_predicted)
            report_file = os.path.join(project_folder, 'classification_report.csv')
            pd.DataFrame(report_df).to_csv(report_file)
        except Exception as e:
            print('Error computing classification report:', e)

    return output_file


def set_args_from_file(args: argparse.Namespace) -> argparse.Namespace:
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


def load_model(args: argparse.Namespace) -> Trainer:
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = AutoModelForSequenceClassification.from_pretrained(
        args.load).to(device)
    trainer = Trainer(model=model)
    return trainer


# MODE = 'train' e.g. python model/model.py --model pubmedbert --data data/prepared_data/asreview_dataset_all --task 'Relevant Sample'
def finetune(args: argparse.Namespace) -> None:
    project_path = init_directories(args.model, args.task)
    save_train_args(project_path, args)
    meta_file = os.path.join(args.data, 'meta.json')
    meta_data = json.load(open(meta_file, 'r'))
    # TODO: solve it nice that meta data is not loaded twice
    train_dataset, test_dataset, val_dataset = load_data(args.data, meta_file)
    trainer = train(project_path, train_dataset, test_dataset,
                    args, val_dataset=val_dataset, is_multilable=meta_data['Is_multilabel'])
    evaluate(project_path, trainer, test_dataset)


#  MODE = 'cont_train', e.g. python model/model.py --mode cont_train --load model/experiments/pubmedbert_relevant_sample_20240730/checkpoint-565
def cont_finetune(args: argparse.Namespace) -> None:
    args = set_args_from_file(args)
    meta_file = os.path.join(args.data, 'meta.json')
    meta_data = json.load(open(meta_file, 'r'))
    train_dataset, test_dataset, eval_dataloader = load_data(
        args.data, meta_file)
    trainer = train(args.load, train_dataset, test_dataset,
                    args, resume_from_checkpoint=True, is_multilable=meta_data['Is_multilabel'])
    evaluate(args.load, trainer, test_dataset)


# MODE = 'eval', e.g. python model/model.py --mode eval --load model/experiments/pubmedbert_relevant_sample_20240730/checkpoint-565
def load_and_evaluate(args: argparse.Namespace) -> None:
    exp_path = os.path.dirname(args.load)
    args = set_args_from_file(args)
    trainer = load_model(args)
    test_dataset = load_data(args.data)[1]
    evaluate(exp_path, trainer, test_dataset)


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


if __name__ == "__main__":
    main()
