import json
import pandas as pd
import os
import requests
from ast import literal_eval
from tqdm import tqdm


STUDIES = '/home/vera/Documents/Arbeit/CRS/PsychNER/app_dash/data/studies.csv'
PREDICTIONS = '/home/vera/Documents/Arbeit/CRS/PsychNER/app_dash/data/predictions.csv'

substance_prediction = '/home/vera/Documents/Arbeit/CRS/PsychNER/model/experiments/pubmedbert_substances_20240902/checkpoint-440_psychedelic_study_relevant_predictions.csv'
condition_prediction = '/home/vera/Documents/Arbeit/CRS/PsychNER/model/experiments/pubmedbert_condition_20240912/checkpoint-792_psychedelic_study_relevant_predictions.csv'
study_information = '/home/vera/Documents/Arbeit/CRS/PsychNER/data/raw_data/asreview_dataset_all_Psychedelic Study.csv'
labels = '/home/vera/Documents/Arbeit/CRS/PsychNER/prodigy/choice_labels.json'
PROD_DIR = '/home/vera/Documents/Arbeit/CRS/PsychNER/'


def read_in_labels(csv: str) -> dict[str, list[str]]:
    with open(csv, 'r') as file:
        data = json.load(file)
    ungrouped_labels = {}
    for d in data:
        for key, value in d.items():
            ungrouped_labels[key] = value
    return ungrouped_labels

def read_in_predictions(csv: str) -> pd.DataFrame:    
    params_file = os.path.join(os.path.dirname(csv), 'params.json')
    with open(params_file, 'r') as file:
        params = json.load(file)
        
    data_path = params['data']
    task = params['task']
    meta_file = os.path.join(PROD_DIR, data_path, f'meta.json')
    with open(meta_file, 'r') as file:
                meta = json.load(file)
                int_to_label = meta['Int_to_label']
                int_to_label = {int(k): v for k, v in int_to_label.items()}
                is_multilabel = meta['Is_multilabel']
    task = task.capitalize()
    
    pred_df = pd.read_csv(csv)
    new_df = []
    # iterate through all rows in the dataframe
    for _, row in pred_df.iterrows():
        for i, prob in enumerate(literal_eval(row['probability'])):
            pred_dict = {
                'id': row['id'],
                'task': task,                
                'label': int_to_label[i],
                'probability': prob,
                'is_multilabel': is_multilabel
            }
            new_df.append(pred_dict)
    
    return pd.DataFrame(new_df)


def read_in_study_information(csv: str) -> pd.DataFrame:
    new_df = []
    study_df = pd.read_csv(csv)
    study_df = study_df[study_df['included'] == 1]
    for _, row in tqdm(study_df.iterrows(), total=study_df.shape[0]):
        # url = row['url']
        # if url is nan
        # if pd.isnull(url):
        #     url = get_url(row['doi'])
        study_dict = {
            'id': row['record_id'],
            'title': row['title'],
            'abstract': row['abstract'],
            'doi': row['doi'],
            'keywords': row['keywords'],
            'year': int(row['year']),
            # 'url': url
        }
        
        new_df.append(study_dict)
    return pd.DataFrame(new_df)
        
def get_url(doi: str) -> str:
    """Get the link to pubmed of the article with the given DOI."""
    # PubMed API endpoint
    pubmed_api_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'

    # Parameters for the PubMed API request
    params = {
        'db': 'pubmed',
        'term': doi,
        'format': 'json'
    }

    try:
        # Send HTTP GET request to PubMed API
        response = requests.get(pubmed_api_url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse JSON response
        data = response.json()

        if data['esearchresult']['idlist']:
            # Extract PubMed ID (PMID) from response
            pmid = data['esearchresult']['idlist'][0]

            # Construct PubMed URL
            pubmed_url = f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/'

            return pubmed_url
        else:
            return ''

    except Exception as e:
        # Handle any exceptions (e.g., network errors, JSON parsing errors)
        print(f"Error occurred: {e}")
        return ''

def main():
    
    task_name1 = 'Substances'
    task_name2 = 'Condition'
    df1 = read_in_predictions(substance_prediction)
    df2 = read_in_predictions(condition_prediction)
    # add the two dataframes together
    df = pd.concat([df1, df2])
    # save the dataframe to a csv file
    df.to_csv(os.path.join('./data', 'predictions.csv'))
    df = read_in_study_information(study_information)
    df.to_csv(os.path.join('./data', 'studies.csv'))
    
    
    
if __name__ == '__main__':
    main()   