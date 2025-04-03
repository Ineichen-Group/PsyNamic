import pandas as pd
import json
import random
import requests
from datetime import datetime
from tqdm import tqdm
import os
import time
from xml.etree import ElementTree

ANNOTATION_DIR = 'prodigy_inputs/annotation_logs/'
PRODIGY_INPUTS_DIR = 'prodigy_inputs/'
ANNOTATION_GROUPS = ['Study Characteristics',
                     'Substance(s)', 'Clinical Measure']

# Script to prepare data for annotation in Prodigy which includes:
# 1. Combining title and abstract into a single text column
# 2. Getting the URL of the article on PubMed
# 3. Generating a random subsample of articles for annotation
# 4. Generating an annotation log to keep track of annotated articles
# 5. Updating the annotation log after each annotation round
# 6. Some more helper functions to manage the annotation process


def get_url(doi: str, title: str='') -> str:
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
    
def get_pubmed_id_from_title(title: str) -> str:
    """
    Searches PubMed for a given article title and returns the corresponding PubMed ID (PMID).

    Args:
        title (str): The title of the article to search for.

    Returns:
        str: The PubMed ID (PMID) if found, otherwise None.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": title,
        "retmode": "xml",
        "retmax": 1  # Get only the first result
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code == 200: 
        root = ElementTree.fromstring(response.content)
        pmid_elem = root.find(".//Id")
        message = root.find(".//OutputMessage")
        if message is not None and message.text == 'No items found.':
            return None
        if pmid_elem is not None:
            pmid = pmid_elem.text
            # Verify the title matches
            fetched_title = fetch_title_from_pubmed_id(pmid)
            if fetched_title and title.lower() == fetched_title.lower():
                return pmid
            else:
                return None
    return None


def fetch_title_from_pubmed_id(pmid: str) -> str:
    """
    Fetches the title of a PubMed article using its PubMed ID.

    Args:
        pmid (str): The PubMed ID of the article.

    Returns:
        str: The title of the article, or None if not found.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml"
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        root = ElementTree.fromstring(response.content)
        title_elem = root.find(".//Item[@Name='Title']")
        if title_elem is not None:
            return title_elem.text
    return None
    
def pubmed_id_valid(pubmed_id: str, titel: str) -> dict:
    """
    Fetches the title of a PubMed article using its PubMed ID.
    
    Args:
        pubmed_id (str or int): The PubMed ID of the article.
    
    Returns:
        str: The title of the article, or None if not found.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {
        "db": "pubmed",
        "id": pubmed_id,
        "retmode": "xml"
    }
    # if pubmed is nan
    if pd.isna(pubmed_id):
        return
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        root = ElementTree.fromstring(response.content)
        title_elem = root.find(".//Item[@Name='Title']")
        if title_elem is not None:
            if title_elem.text == titel:
                return True
            elif titel in title_elem.text:
                return True
            else:
                return False
    else:
        return False


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:

    # Fill missing values with empty strings
    df.fillna('', inplace=True)
    # Get the URL of the article on PubMed
    tqdm.pandas()
    df['pubmed_url'] = df['doi'].progress_apply(lambda x: get_url(x))
    # Combine title and abstract into a single text column
    df['text'] = df['title'] + '.^\n' + df['abstract']
    return df


def generate_subsample(data_file: str, n: int, annotation_log: bool = True, random_seed=42) -> str:
    """ Generate a random subsample of n articles from the given DataFrame and save it as a JSONL file."""
    random.seed(random_seed)
    df = pd.read_csv(data_file)
    relevant_cols = ['record_id',  'keywords', 'text',
                     'doi', 'pubmed_url', 'secondary_title']
    # remove nan
    df = df.fillna('')
    data = df[relevant_cols].to_dict(orient='records')

    if annotation_log:
        log_df = _get_most_recent_annotation_log()
        # Exclude samples were annotated equal to True
        exclude = log_df[log_df['annotated'] == True]['record_id'].tolist()
        data = [d for d in data if d['record_id'] not in exclude]
    random_subset = random.sample(data, n)
    current_date = datetime.now().strftime('%Y%m%d')
    current_time = datetime.now().strftime('%H%M%S')
    output_file = f'{
        PRODIGY_INPUTS_DIR}/psychedelic_study_{n}_{current_date}_{current_time}.jsonl'

    with open(output_file, 'w', encoding='utf-8') as f:
        for d in random_subset:
            for group in ANNOTATION_GROUPS:
                d_new = d.copy()
                d_new['annotation'] = group
                f.write(json.dumps(d_new, ensure_ascii=False) + '\n')
    time.sleep(1)
    return output_file


def generate_annotation_log(annotation_log: str, raw_data: str) -> None:
    df = pd.read_csv(raw_data)
    # Check if there are duplicates in record_id
    if df['record_id'].duplicated().any():
        raise ValueError('Duplicates in record_id column')
    log = pd.DataFrame(
        {'record_id': df['record_id'], 'annotated': False, 'data_set': ''})
    log.to_csv(annotation_log, index=False)


def update_annotation_log(sample_file: str) -> None:
    log_df = _get_most_recent_annotation_log()
    log_df['data_set'] = log_df['data_set'].astype(str)
    log_df['data_set'] = log_df['data_set'].replace('nan', '')
    with open(sample_file, 'r') as infile:
        doubled = []
        for line in infile:
            record_id = json.loads(line)['record_id']
            # set annotated to True and add sample file name
            sample_file_name = os.path.basename(sample_file)

            # check if record_id is already annotated
            already_annotated = log_df.loc[log_df['record_id']
                                           == record_id, 'annotated'].values[0]
            dataset = log_df.loc[log_df['record_id']
                                 == record_id, 'data_set'].values[0]
            if already_annotated and dataset != sample_file_name:
                print(f'{record_id} already annotated by {dataset}')
                doubled.append(record_id)
            else:
                log_df.loc[log_df['record_id'] == record_id,
                           'data_set'] = sample_file_name
                log_df.loc[log_df['record_id'] ==
                           record_id, 'annotated'] = True
        # write doubled to file
        if doubled:
            doubled = set(doubled)
            with open('doubled.txt', 'a') as f:
                for record_id in doubled:
                    f.write(str(record_id) + '\n')

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


def remove_from_annotation_log(subsample: str) -> None:
    log_df = _get_most_recent_annotation_log()
    rows_index = log_df.index[log_df['data_set'] == subsample]
    log_df.loc[rows_index, 'annotated'] = False
    log_df.loc[rows_index, 'data_set'] = ''

    current_date = datetime.now().strftime('%Y%m%d')
    new_log = f'annotation_log_{current_date}.csv'
    new_log_path = os.path.join(ANNOTATION_DIR, new_log)
    log_df.to_csv(new_log_path, index=False)


def readd_partial_annotation_log(partial_annotated: str) -> None:
    log_df = _get_most_recent_annotation_log()
    file_name = os.path.basename(partial_annotated)
    # extract 20240423_113436 from prodigy_export_bernard_200_20240423_113436_20240713_171907.jsonl
    file_identifier = file_name.split('_')[4] + '_' + file_name.split('_')[5]
    # read in annotated records into a annoated_df
    annot_ct = 0
    not_annot_ct = 0
    with open(partial_annotated, 'r') as infile:
        annotated_df = pd.DataFrame([json.loads(line) for line in infile])
        # iterate through log_df and check if record_id was annotated
        for index, row in log_df.iterrows():
            if file_identifier in str(row['data_set']):
                record_id = row['record_id']
                # if record_id not annotated, set annotated to False and remove data_set
                if record_id not in annotated_df['record_id'].values:
                    log_df.loc[index, 'annotated'] = False
                    log_df.loc[index, 'data_set'] = ''
                    not_annot_ct += 1
                else:
                    annot_ct += 1
    print(f'Annotated: {annot_ct}, Not Annotated & removed from log: {
          not_annot_ct}')

    # write new log
    current_date = datetime.now().strftime('%Y%m%d')
    new_log = f'annotation_log_{current_date}.csv'
    new_log_path = os.path.join(ANNOTATION_DIR, new_log)
    log_df.to_csv(new_log_path, index=False)


def annotation_progress() -> None:
    log_df = _get_most_recent_annotation_log()
    total = log_df.shape[0]
    annotated = log_df['annotated'].sum()
    print(f'Annotated: {annotated}/{total}')


def add_missing_annotation_log() -> None:
    all_studies = 'raw_data/asreview_dataset_all_Psychedelic Study.csv'
    all_studies_df = pd.read_csv(all_studies)

    log_df = _get_most_recent_annotation_log()
    all_studies_df = all_studies_df[all_studies_df['included'] == 1]
    add_data = []

    for _, row in all_studies_df.iterrows():
        record_id = row['record_id']
        if record_id not in log_df['record_id'].values:
            add_data.append(row)
    new_data_df = prepare_data(pd.DataFrame(add_data))

    # append new data to all_studies_df
    all_cleaned = 'raw_data/dataset_relevant_cleaned.csv'
    all_cleaned_df = pd.read_csv(all_cleaned)
    all_cleaned_df = pd.concat([all_cleaned_df, new_data_df])
    all_cleaned_df.to_csv(all_cleaned, index=False)

    # add to annotation log
    new_data_df['annotated'] = False
    new_data_df['data_set'] = ''
    new_data_df = new_data_df[['record_id', 'annotated', 'data_set']]
    log_df = pd.concat([log_df, new_data_df])

    # write new log
    current_date = datetime.now().strftime('%Y%m%d')
    new_log = f'annotation_log_{current_date}.csv'
    new_log_path = os.path.join(ANNOTATION_DIR, new_log)
    log_df.to_csv(new_log_path, index=False)

def extract_pubmed_id(csv_file: str) -> None:

    def pubmed_url_to_id(url: str):
        # if nan return empty string
        if pd.isna(url):
            return ''
        id = url.split('/')[-2]
        return id

    with open(csv_file, 'r', encoding='utf-8') as infile:
        df = pd.read_csv(infile)
        df['pubmed_id'] = df['pubmed_url'].apply(lambda x: pubmed_url_to_id(x))
        df.to_csv(csv_file, index=False)

def fix_pubmed_id(csv_file: str) -> None:
    outdata = []
    with open(csv_file, 'r', encoding='utf-8') as infile:
        df = pd.read_csv(infile)
        for index, row in df.iterrows():
            print(f'Processing {index}')
            if pubmed_id_valid(row['pubmed_id'], row['title']):
                outdata.append(row)
            else:
                pubmed_id = get_pubmed_id_from_title(row['title'])
                if not pubmed_id:
                    row['pubmed_id'] = ''
                    row['pubmed_url'] = ''
                else:
                    row['pubmed_id'] = pubmed_id
                    row['pubmed_url'] = f'https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/'
                outdata.append(row)
    out_df = pd.DataFrame(outdata)
    out_df.to_csv(csv_file, index=False)
            

def main():

    # Load the raw data and prepare it
    raw_data = 'raw_data/asreview_dataset_relevant_Psychedelic Study.csv'
    cleaned_data = 'raw_data/dataset_relevant_cleaned.csv'
    annotation_log = 'prodigy_inputs/annotation_logs/annotation_log.csv'

    # df = pd.read_csv(raw_data)
    # prepare_df = prepare_data(df)
    # prepare_df.to_csv('raw_data/dataset_relevant_cleaned.csv', index=False)

    # Generate annotation log: record_id, annotated
    # generate_annotation_log(annotation_log, raw_data)

    # # Add 100 samples from V3 to annotation log
    # new_annotated = 'prodigy_inputs/psychedelic_study_100_20240411.jsonl'
    # update_annotation_log(new_annotated)
    # annotation_progress()

    # # Generate subsample of 50 and 100 samples
    # subsample_file = generate_subsample(cleaned_data, 50)
    # update_annotation_log(subsample_file)
    # annotation_progress()

    # subsample_file = generate_subsample(cleaned_data, 100)
    # update_annotation_log(subsample_file)
    # annotation_progress()
    # subsample = generate_subsample(cleaned_data, 250)
    # update_annotation_log(subsample)
    # subsample = generate_subsample(cleaned_data, 250)
    # update_annotation_log(subsample)
    # subsample = generate_subsample(cleaned_data, 250)
    # update_annotation_log(subsample)
    # subsample = generate_subsample(cleaned_data, 250)
    # update_annotation_log(subsample)

    # remove julia, bernard, pia
    # annotation_progress()
    # remove_from_annotation_log('psychedelic_study_250_20240423_113435.jsonl')
    # annotation_progress()
    # remove_from_annotation_log('psychedelic_study_250_20240423_113436.jsonl')
    # annotation_progress()
    # remove_from_annotation_log('psychedelic_study_250_20240423_113437.jsonl')
    # annotation_progress()

    # subsample = generate_subsample(cleaned_data, 250)
    # update_annotation_log(subsample)
    # annotation_progress()

    # subsample = generate_subsample(cleaned_data, 250)
    # update_annotation_log(subsample)
    # annotation_progress()

    # subsample = generate_subsample(cleaned_data, 250)
    # update_annotation_log(subsample)
    # annotation_progress()

    # subsample = generate_subsample(cleaned_data, 250)
    # update_annotation_log(subsample)
    # annotation_progress()

    # remove_from_annotation_log('psychedelic_study_250_20240425_134820.jsonl')
    # remove_from_annotation_log('psychedelic_study_250_20240425_134821.jsonl')
    # remove_from_annotation_log('psychedelic_study_250_20240425_134822.jsonl')
    # remove_from_annotation_log('psychedelic_study_250_20240425_134823.jsonl')
    # remove_from_annotation_log('psychedelic_study_40_20240523_162946.jsonl')
    # remove_from_annotation_log('psychedelic_study_250_20240425_152801.jsonl')

    # update_annotation_log('prodigy_inputs/psychedelic_study_250_20240423_113435.jsonl')
    # update_annotation_log('prodigy_inputs/psychedelic_study_250_20240423_113436.jsonl')
    # update_annotation_log('prodigy_inputs/psychedelic_study_250_20240423_113437.jsonl')
    # update_annotation_log('prodigy_inputs/psychedelic_study_95_20240423_113434.jsonl')
    # update_annotation_log('prodigy_inputs/psychedelic_study_24_20240425_152801.jsonl')
    # annotation_progress()

    # subsample = generate_subsample(cleaned_data, 40)
    # update_annotation_log(subsample)
    # annotation_progress()

    # subsample = generate_subsample(cleaned_data, 250)
    # update_annotation_log(subsample)
    # annotation_progress()

    # subsample = generate_subsample(cleaned_data, 40)
    # update_annotation_log(subsample)
    # annotation_progress()

    # file = 'prodigy_exports/prodigy_export_bernard_250_20240423_113436_20240713_171907.jsonl'
    # readd_partial_annotation_log(file)
    # annotation_progress()

    # subsample = generate_subsample(cleaned_data, 250)
    # update_annotation_log(subsample)
    # annotation_progress()

    # readd_partial_annotation_log(
    #     'prodigy_exports/prodigy_export_ben_120_20240523_195806_20241206_095404.jsonl')
    # readd_partial_annotation_log(
    #     'prodigy_exports/prodigy_export_julia_110_20240423_113435_20240812_012727.jsonl')
    # annotation_progress()

    # add_missing_annotation_log()
    # extract_pubmed_id('data/raw_data/dataset_relevant_cleaned.csv')
    fix_pubmed_id('data/raw_data/dataset_relevant_cleaned.csv')

if __name__ == '__main__':

    main()
