import pandas as pd
import json
import random
import requests
from datetime import datetime
from tqdm import tqdm
import os

ANNOTATION_DIR = 'prodigy_inputs/annotation_logs/'
PRODIGY_INPUTS_DIR = 'prodigy_inputs/'


def get_url(doi: str) -> str:
    """Get the link to pubmed of the article with the given DOI."""
    if not doi:
        return ''
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


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:

    # Fill missing values with empty strings
    df.fillna('', inplace=True)
    # Get the URL of the article on PubMed
    tqdm.pandas()
    df['pubmed_url'] = df['doi'].progress_apply(lambda x: get_url(x))
    # Combine title and abstract into a single text column
    df['text'] = df['title'] + '.^\n' + df['abstract']
    return df


def generate_subsample(data_file: str, n: int, annotation_log: bool = True) -> str:
    """ Generate a random subsample of n articles from the given DataFrame and save it as a JSONL file."""
    df = pd.read_csv(data_file)
    relevant_cols = ['record_id',  'keywords', 'text',
                     'doi', 'pubmed_url', 'secondary_title']
    data = df[relevant_cols].to_dict(orient='records')

    if annotation_log:
        log_df = _get_most_recent_annotation_log()
        # Exclude samples were annotated equal to True
        exclude = log_df[log_df['annotated'] == True]['record_id'].tolist()
        data = [d for d in data if d['record_id'] not in exclude]
    random_subset = random.sample(data, n)
    current_date = datetime.now().strftime('%Y%m%d')
    output_file = f'{PRODIGY_INPUTS_DIR}/psychedelic_study_{n}_{current_date}.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for d in random_subset:
            f.write(json.dumps(d) + '\n')

    return output_file


def generate_annotation_log(annotation_log: str, raw_data: str) -> None:
    df = pd.read_csv(raw_data)
    # Check if there are duplicates in record_id
    if df['record_id'].duplicated().any():
        raise ValueError('Duplicates in record_id column')
    log = pd.DataFrame({'record_id': df['record_id'], 'annotated': False})
    log.to_csv(annotation_log, index=False)


def update_annotation_log(sample_file: str) -> None:
    log_df = _get_most_recent_annotation_log()
    with open(sample_file, 'r') as infile:
        for line in infile:
            record_id = json.loads(line)['record_id']
            # set annotated to True
            log_df.loc[log_df['record_id'] == record_id, 'annotated'] = True

    current_date = datetime.now().strftime('%Y%m%d')
    new_log = f'annotation_log_{current_date}.csv'
    new_log_path = os.path.join(ANNOTATION_DIR, new_log)
    log_df.to_csv(new_log_path, index=False)


def _get_most_recent_annotation_log() -> pd.DataFrame:
    # get latest annotation log within directory:
    annotation_log = sorted(
        [f for f in os.listdir(ANNOTATION_DIR) if f.startswith('annotation_log')])[-1]
    annotation_log_path = os.path.join(ANNOTATION_DIR, annotation_log)
    return pd.read_csv(annotation_log_path)


def annotation_progress() -> None:
    log_df = _get_most_recent_annotation_log()
    total = log_df.shape[0]
    annotated = log_df['annotated'].sum()
    print(f'Annotated: {annotated}/{total}')

def main():

    # Load the raw data and prepare it
    # raw_data = 'raw_data/asreview_dataset_relevant_Psychedelic Study.csv'
    # df = pd.read_csv(raw_data)
    # prepare_df = prepare_data(df)
    # prepare_df.to_csv('raw_data/dataset_relevant_cleaned.csv', index=False)

    # Generate annotation log: record_id, annotated
    # annotation_log = 'prodigy_inputs/annotation_logs/annotation_log.csv'
    # generate_annotation_log(annotation_log, raw_data)
    new_annotated = 'prodigy_inputs/psychedelic_study_100_20240411.jsonl'
    update_annotation_log(new_annotated)
    cleaned_data = 'raw_data/dataset_relevant_cleaned.csv'
    subsample_file = generate_subsample(cleaned_data, 50)
    update_annotation_log(subsample_file)
    annotation_progress()


if __name__ == '__main__':
    
    main()
