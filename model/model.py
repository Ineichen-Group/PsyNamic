from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback, AdamW, get_linear_schedule_with_warmup
from datahandler import DataHandler, PsychNamicRelevant, PsyNamicSingleLabel, DataSplit
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
from datetime import datetime

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

# os.environ['WANDB_PROJECT'] = "psynamic"
# os.environ['WANDB_LOG_MODEL'] = "checkpoint"
os.environ['WANDV_MODE'] = 'offline'
os.environ['WANDB_DISABLED'] = 'true'

# TODO: Enable using val dataset
# TODO: Enable cross validation
# TODO: enable all hyperparameters
# TODO: Hyperparameter tune


def init_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)

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

    # Training
    parser.add_argument('--cross_val', type=bool, default=False)

    # Evaluation
    parser.add_argument(
        '--metrics', type=list[str], default=['accuracy', 'precision', 'recall', 'f1'])
    return parser.parse_args()


def init_directories(model: str, task: str) -> str:
    date = datetime.now().strftime("%Y%m%d")
    lower_case_task = task.lower().replace(' ', '_')
    experiment_path = f'{model}_{lower_case_task}_{date}'
    project_dir = os.path.join(EXPERIMENT_PATH, experiment_path)
    os.makedirs(project_dir, exist_ok=True)
    return project_dir


def load_data(data: str):
    meta_file = os.path.join(data, 'meta.json')
    datahandler = DataHandler(data, meta_file=meta_file)
    datahandler.load_splits(data)
    train_dataset, test_dataset, eval_dataset = datahandler.get_strat_split()
    return train_dataset, test_dataset, eval_dataset


def train(project_dir, train_dataset, test_dataset, args) -> None:
    model_id = MODEL_IDENTIFIER[args.model]
    # Initialize the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_id)
    model = BertForSequenceClassification.from_pretrained(
        model_id, num_labels=train_dataset.nr_labels)
    print(f'Length of train dataset: {len(train_dataset)}')
    print(f'Lenght of test dataset: {len(test_dataset)}')

    training_args = TrainingArguments(
        output_dir=project_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        # logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        # report_to='wandb',
        report_to='none',
        logging_dir=project_dir,
        logging_strategy="epoch",
        run_name=os.path.basename(project_dir)
    )
    total_steps = len(train_dataset) * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.learning_rate,
                      weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(training_args.warmup_ratio * total_steps), num_training_steps=total_steps)

    # Define compute metrics function
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        # Calculate accuracy
        accuracy = accuracy_score(labels, preds)

        # Calculate precision, recall, and F1-score
        precision = precision_score(labels, preds, average='weighted')
        recall = recall_score(labels, preds, average='weighted')
        f1 = f1_score(labels, preds, average='weighted')

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
    )

    if args.early_stopping_patience > 0:
        training_args.load_best_model_at_end = True
        training_args.metric_for_best_model = 'eval_loss'
        trainer.callbacks = [EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience)]

    # Train the model
    trainer.train()
    return trainer


def evaluate(project_folder, trainer: Trainer, test_dataset: DataSplit) -> None:
    # Write id, text, true_label, pred_label and probabilities to file
    pass
   


def main():
    args = init_argparse()
    project_path = init_directories(args.model, args.task)
    train_dataset, test_dataset, eval_dataloader = load_data(args.data)
    trainer = train(project_path, train_dataset, test_dataset, args)
    evaluate(project_path, trainer)


if __name__ == "__main__":
    main()
