
import pandas as pd

data = 'data/prepared_data/psychedelic_study_relevant.csv'
ids = 'data/prepared_data/round2_ids.txt'

df = pd.read_csv(data)

# read ids
with open(ids, 'r', encoding='utf-8') as f:
    ids = f.readlines()
ids = [int(x.strip()) for x in ids]

to_be_annotated = []

for index, row in df.iterrows():
    if row['id'] in ids:
        continue
    else:
        to_be_annotated.append(row)

df = pd.DataFrame(to_be_annotated)

df.to_csv('data/prediction_data/psychedelic_study_unannoated.csv', index=False)
