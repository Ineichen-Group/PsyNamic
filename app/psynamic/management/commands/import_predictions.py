import json
import os
from ast import literal_eval

import pandas as pd
from django.core.management.base import BaseCommand, CommandParser
from psynamic.models import LabelClass, Prediction, Study
from tqdm import tqdm

# python manage.py import_predictions /home/vera/Documents/Arbeit/CRS/PsychNER/model/experiments/pubmedbert_condition_20240912/checkpoint-792_psychedelic_study_relevant_predictions.csv /home/vera/Documents/Arbeit/CRS/PsychNER/model/experiments/pubmedbert_condition_20240912/params.json

# python manage.py import_predictions /home/vera/Documents/Arbeit/CRS/PsychNER/model/experiments/pubmedbert_substances_20240902/checkpoint-440_psychedelic_study_relevant_predictions.csv /home/vera/Documents/Arbeit/CRS/PsychNER/model/experiments/pubmedbert_substances_20240902/params.json

class Command(BaseCommand):
    help = 'Import predictions from a CSV file into the database'

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument('pred_csv', type=str)
        parser.add_argument('params_file', type=str)

    def handle(self, *args, **kwargs):
        pred_df = pd.read_csv(kwargs['pred_csv'])
        with open(kwargs['params_file'], 'r') as file:
            params = json.load(file)
            
        task = params['task']
        data_path = params['data']
        model = params['model']
        
        # get meta json in data path
        # TODO: don't hardcode path
        meta_file = os.path.join('/home/vera/Documents/Arbeit/CRS/PsychNER/', data_path, f'meta.json')
        with open(meta_file, 'r') as file:
            meta = json.load(file)
            int_to_label = meta['Int_to_label']
            int_to_label = {int(k): v for k, v in int_to_label.items()}
            is_multilabel = meta['Is_multilabel']
        task = task.capitalize()
        label_class = LabelClass.objects.get(name=task)
        labels = label_class.labels.all()
        label_class.is_multilabel = is_multilabel
        label_class.save()
        for _, row in tqdm(pred_df.iterrows(), total=pred_df.shape[0], desc="Importing predictions"):
            # check if study already exists
            study_id = row['id']
            # get or create study
            try:
                study = Study.objects.get(id=study_id)
            except Study.DoesNotExist:
                study = Study.objects.create(
                    id=study_id,
                    text=row['text']
                )
            probabilities = literal_eval(row['probability'])
            for i, prob in enumerate(probabilities):
                label = labels.get(name=int_to_label[i])
                pred = Prediction.objects.create(
                    label=label,
                    model_name=model,
                    probability=prob
                )
                study.predictions.add(pred)
            study.save()