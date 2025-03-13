import datetime
import os
import pandas as pd
import json
from prodigy_data_reader import FIXED_COLUMNS


# Prepare annotated data for the webapp
def prepare_annotated_data():
    DATA_PATH = 'prepared_data/training_round2/'
    OUTPUT_PATH = '../PsyNamic-Webapp/data/'

    today = datetime.datetime.now().strftime('%Y-%m-%d')
    OUTPUT_FILE = os.path.join(OUTPUT_PATH, f'predictions_{today}.csv')

    default_prob = 1.0
    default_model = 'manual_annotation'

    output_data = []
    for file in os.listdir(DATA_PATH):
        if file.endswith('.csv'):
            print(f'Processing {file}')
            data = pd.read_csv(os.path.join(DATA_PATH, file))
            meta_file = file.split('.')[0] + '_meta.json'
            meta_data = json.load(open(os.path.join(DATA_PATH, meta_file)))
            task = meta_data['Task']
        
            if meta_data['Is_multilabel'] and file.startswith('onehot'):
                label_cols = [col for col in data.columns if col not in FIXED_COLUMNS]

                for i, row in data.iterrows():
                    for label_col in label_cols:
                        if row[label_col] == 1:
                            output_line = {
                                'id': row['id'],
                                'task': task,
                                'label': label_col,
                                'probability': default_prob,
                                'model': default_model,
                                'is_multilabel': True
                            }
                            output_data.append(output_line)
            elif not meta_data['Is_multilabel'] and not file.startswith('onehot'):
                int2label = {int(k): v for k, v in meta_data['Int_to_label'].items()}
                for i, row in data.iterrows():
                    output_line = {
                        'id': row['id'],
                        'task': task,
                        'label': int2label[row[task]],
                        'probability': default_prob,
                        'model': default_model,
                        'is_multilabel': False
                    }
                    output_data.append(output_line)
            else:
                raise ValueError('Data format not supported')

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f'Output saved to {OUTPUT_FILE}')

def main():
    prepare_annotated_data()
