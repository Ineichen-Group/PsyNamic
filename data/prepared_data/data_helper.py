
import pandas as pd


data_prepared = 'data/prepared_data/psychedelic_study_relevant.csv'
raw_data = 'data/raw_data/dataset_relevant_cleaned.csv'
annotated_ids = 'data/prepared_data/round2_ids.txt'

outfile = 'data/prepared_data/psynamic_relevant_with_info_03042025.csv'

relevant_columns = [
    'keywords',
    'year',
    'title',
    'url',
    'abstract',
    'doi',
    'pubmed_url',
    'authors',
]



with open(data_prepared, 'r', encoding='utf-8') as f:
    df = pd.read_csv(f, encoding='utf-8')

    df_raw = pd.read_csv(raw_data, encoding='utf-8')

    annotated_ids = [int(line.strip()) for line in open(annotated_ids, 'r', encoding='utf-8').readlines()]
    df['annotated_manually'] = 0
    df.loc[df['id'].isin(annotated_ids), 'annotated_manually'] = 1

    for id in df['id']:
        row = df_raw[df_raw['record_id'] == id]
        # check if length of row is 1
        if len(row) != 1:
            breakpoint()
        for col in relevant_columns:
                df.loc[df['id'] == id, col] = row[col].values[0]
        
    df = df[['id', 'title', 'doi', 'pubmed_url', 'url', 'abstract', 'authors', 'year', 'keywords', 'annotated_manually']]

    # save the dataframe to a csv file
    df.to_csv(outfile, index=False, encoding='utf-8')

