import pandas as pd
import json

# Script used to extract the dosages from the NER data for Ben to devise normalisation rules

infile = '/home/vera/Documents/Arbeit/CRS/PsychNER/data/prepared_data/training_round2/ner_bio_966.jsonl'

df = pd.read_json(infile, lines=True)

out_lines = []
# iterate through the rows
for i, row in df.iterrows():
    dosage = ''
    tokens = row['tokens']
    labels = row['ner_tags']
    id = row['id']    
    
    line_dict = {}
    for i, (t, l) in enumerate(zip(tokens, labels)):
        if l == 'B-Dosage':
            start_i = i
            dosage = t + ' '
        elif l == 'I-Dosage':
            dosage += t + ' '
        elif l == 'O' or i == len(tokens) - 1:
            end_i = i
            if dosage:
                print(id, dosage)
                line_dict['dosage'] = dosage.strip()
                line_dict['id'] = id
                # add 5 tokens before and after
                line_dict['context'] = ' '.join(tokens[max(0, start_i-5):min(len(tokens), end_i+5)])
                line_dict['text']  = ' '.join(tokens)
                out_lines.append(line_dict)
                dosage = ''     
                line_dict = {}

outfile = 'data/dosages.csv'
# wrote to csv, make sure newlines are treated as newlines
pd.DataFrame(out_lines).to_csv(outfile, index=False, sep='\t')
