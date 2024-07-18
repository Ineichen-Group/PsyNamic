from data.prodigy_data_reader import ProdigyDataCollector
from datahandler import PsyNamicSingleLabel, PsychNamicRelevant
import os
import time
import json

# TODO: add meta file to directory with splits (containing split information, task, file, int_to_label)

def prepare_train_data(list_jsonl: str):
    prodigy_data = ProdigyDataCollector(list_jsonl)
    tasks = prodigy_data.tasks.keys()
    prepared_data = 'data/prepared_data/'
    for task in tasks:
        task_string = task.replace(' ', '_').lower()
        if prodigy_data.is_multilabel(task):
            task_df = prodigy_data.get_onehot_task_df(task)
            task_df.to_csv(os.path.join(prepared_data, f'onehot_{task_string}.csv'), index=False)

        else:
            int_to_label, task_df = prodigy_data.get_label_task_df(task)
            task_df.to_csv(os.path.join(prepared_data, f'{task_string}.csv'), index=False)
            date = time.strftime("%Y%m%d")    
            meta_data = {
                "Task": task,
                "Files": list_jsonl,
                "Int_to_label": int_to_label
            }
            meta_file = os.path.join(prepared_data, f'{task_string}_meta_{date}.csv')
            with open(meta_file, 'w') as f:
                json.dump(meta_data, f, indent=4, ensure_ascii=False)

def prepare_splits():
    relevant_task = 'Number of Participants'
    file = 'data/prepared_data/number_of_participants.csv'
    meta = 'data/prepared_data/number_of_participants_meta_20240709.csv'
    data_handler = PsyNamicSingleLabel(file, relevant_task, meta)
    data_handler.get_strat_split()
    data_handler.save_split(f'data/prepared_data/{relevant_task.replace(" ", "_").lower()}/')
    
    file = 'data/raw_data/asreview_dataset_all_Psychedelic Study.csv'
    data_handler = PsychNamicRelevant(file, 'record_id', 'title', 'abstract', 'included')
    data_handler.get_strat_split()
    data_handler.save_split(f'data/prepared_data/asreview_dataset_all/')

# if main equal name
if __name__ == '__main__':
    list_jsonl = [
        'data/prodigy_exports/prodigy_export_ben_95_20240423_113434.jsonl',
        'data/iaa/iaa_round1_50/iaa_resolution/prodigy_export_review_all_text_50_20240418_20240607_145354.jsonl',
        'data/prodigy_exports/prodigy_export_ben_24_20240425_152801.jsonl',
        'data/iaa/iaa_round2_40/iaa_resolution/prodigy_export_review_all_text_40_20240523_20240705_183405.jsonl',
    ]
    prepare_train_data(list_jsonl)
    prepare_splits()