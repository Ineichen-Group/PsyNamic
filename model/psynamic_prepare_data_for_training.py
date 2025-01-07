from data.prodigy_data_reader import ProdigyDataCollector, FIXED_COLUMNS
from datahandler import PsyNamicSingleLabel, PsychNamicRelevant, PsyNamicMultiLabel, DataHandlerBIO
import os
import time
import json
import pandas as pd
from stride_utils.prodigy import ner_process_file_and_save_to_bio_format
import re


def prepare_train_data(prodigy_data: ProdigyDataCollector, outpath: str) -> None:
    if not outpath:
        outpath = 'data/prepared_data'
    tasks = prodigy_data.tasks.keys()
    for task in tasks:
        task_string = task.replace(' ', '_').lower()
        date = time.strftime("%Y%m%d")
        meta_data = {
            "Date": date,
            "Task": task,
            "Files": prodigy_data.prodigy_files,
            "Is_multilabel": prodigy_data.is_multilabel(task),
            "Nr_unnotated (removed)": prodigy_data.get_unannotated(task)
        }
        if prodigy_data.is_multilabel(task):
            meta_file = os.path.join(
                outpath, f'onehot_{task_string}_meta.json')
            task_df = prodigy_data.get_onehot_task_df(task)
            meta_data['Size'] = len(task_df)
            task_df.to_csv(os.path.join(
                outpath, f'onehot_{task_string}.csv'), index=False)
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(meta_data, f, indent=4, ensure_ascii=False)

        else:
            meta_file = os.path.join(
                outpath, f'{task_string}_meta.json')
            label_to_int, task_df = prodigy_data.get_label_task_df(task)
            meta_data['Size'] = len(task_df)
            task_df.to_csv(os.path.join(
                outpath, f'{task_string}.csv'), index=False)

            meta_data["Int_to_label"] = {v: k for k, v in label_to_int.items()}
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(meta_data, f, indent=4, ensure_ascii=False)


def find_file_in_dir(file_string: str, dir: str) -> str:
    for file in os.listdir(dir):
        if file_string in file:
            return os.path.join(dir, file)
    return None


def prepare_bio_data(list_jsonl: list[str], id_field: str, outfile: str, purposes: list[str], expert_annot: str = '') -> str:
    outfile_path = os.path.dirname(outfile)

    if not purposes:
        purposes = ['both'] * len(list_jsonl)

    line_count = 0
    rejected_count = 0
    ids_index_dict = {}
    lines = []

    for file, purpose in zip(list_jsonl, purposes):
        with open(file, 'r', encoding='utf-8') as f:
            # Only use files with ner purpose
            if purpose == 'class':
                continue
            for line in f:
                parsed_line = json.loads(line)
                # Remove rejected answers
                if parsed_line['answer'] == 'accept':
                    id = parsed_line['record_id']
                    # Remove duplciates (from thematic split) and either always prefer first or expert annotator
                    if id not in ids_index_dict:
                        lines.append(line)
                        ids_index_dict[id] = line_count
                        line_count += 1

                    else:
                        if expert_annot in file and expert_annot:
                            # Case 2: id is in ids_index_dict and expert annotator
                            # replace line
                            lines[ids_index_dict[id]] = line
                            line_count += 1
                else:
                    rejected_count += 1

    def get_text_from_span(tokens, start_id, end_id):
        text = ''
        for i in range(start_id, end_id+1):
            for token in tokens:
                if token['id'] == i:
                    text += token['text']
                    if token['ws'] == True:
                        text += ' '
        return text

    def is_whitespace(token:str) -> bool:
        pattern = r'[^\S\r\n]+'
        return re.fullmatch(pattern, token)
    
    def find_whitespace_indices(tokens: dict[str]) -> list[int]:
        """should catch \u00a0 and \u2009 ect."""
        # Match any whitespace except \n, \r
        
        whitespace_ids = [t['id'] for t in tokens if is_whitespace(t['text'])]
        return whitespace_ids

    # fix issues with \u00a0 # TODO: very ugly code but works for now
    new_lines = []
    for line in lines:
        parsed_line = json.loads(line)

        # Case 1: no spans
        if 'spans' not in line:
            new_lines.append(json.dumps(parsed_line, ensure_ascii=False))
            continue

        # Case 2: spans is empty
        elif not parsed_line['spans']:
            new_lines.append(json.dumps(parsed_line, ensure_ascii=False))
            continue

        # Case 3: there are spans
        else:
            spans = parsed_line['spans']
            tokens = parsed_line['tokens']
            ids_ws = find_whitespace_indices(tokens)
            new_spans = []

            # Iterate through spans
            prev_span = spans[0]
            for span in spans[1:]:
                new_spans.append(prev_span)
                # Case 1: no NER tags or different NER tags
                if 'label' not in span or 'label' not in prev_span or span['label'] != prev_span['label']:
                    prev_span = span
                # Case 2: same NER tags
                else:
                    id_between_token = span['token_start'] - 1
                    # Case 1: there is a whitespace between the tokens --> merge
                    if span['token_start'] - prev_span['token_end'] == 2 and id_between_token in ids_ws:
                        # Print merge text
                        prev_span_text = get_text_from_span(
                            tokens, prev_span['token_start'], prev_span['token_end'])
                        span_text = get_text_from_span(
                            tokens, span['token_start'], span['token_end'])
                        new_token_text = get_text_from_span(
                            tokens, prev_span['token_start'], span['token_end'])
                        print(f'"{prev_span_text}" + "{span_text.strip()}" --> "{new_token_text.strip()}"')
                        new_span = {
                            "start": prev_span['start'],
                            "end": span['end'],
                            "token_start": prev_span['token_start'],
                            "token_end": span['token_end'],
                            "label": span['label'],
                        }
                        # Remove last span, add new span
                        new_spans.pop()
                        prev_span = new_span
                    else:
                        prev_span = span

                if span == spans[-1]:
                    new_spans.append(prev_span)

        parsed_line['spans'] = new_spans
        new_lines.append(json.dumps(parsed_line, ensure_ascii=False))

    # Save file
    with open(outfile, 'w', encoding='utf-8') as out:
        for line in new_lines:
            out.write(line + '\n')

    meta_data = {
        "Date": time.strftime("%Y%m%d"),
        "Files": list_jsonl,
        "Size": line_count-rejected_count,
        "Rejected": rejected_count
    }

    with open(outfile.replace('.jsonl', '_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, indent=4, ensure_ascii=False)

    outfile_bio = ner_process_file_and_save_to_bio_format(
        outfile, outfile.replace('.jsonl', '_bio'), id_field)

    # open bio file and remove all \u00a0
    with open(outfile_bio, 'r', encoding='utf-8') as f:
        new_lines = [
            json.dumps(
                {
                    **json.loads(line),
                    'tokens': [token for token in json.loads(line)['tokens'] if not is_whitespace(token)],
                    'ner_tags': [ner for ner, token in zip(json.loads(line)['ner_tags'], json.loads(line)['tokens']) if not is_whitespace(token)]
                },
                ensure_ascii=False
            )
            for line in f
        ]

    with open(outfile_bio, 'w', encoding='utf-8') as f:
        for line in new_lines:
            f.write(line + '\n')

    bio_datahandler = DataHandlerBIO(outfile_bio)
    print(bio_datahandler.label2id)
    bio_datahandler.get_split(use_val=True)
    # get parent folder of outfile
    bio_datahandler.save_split(os.path.join(
        os.path.dirname(outfile), 'ner_bio/'))


def prepare_splits(data_path: str):

    tasks = [
        "Data Collection",
        "Data Type",
        "Number of Participants",
        "Age of Participants",
        "Application Form",
        "Clinical Trial Phase",
        "Condition",
        "Outcomes",
        "Regimen",
        "Setting",
        "Study Control",
        "Study Purpose",
        "Substance Naivety",
        "Substances",
        "Sex of Participants",
        "Study Conclusion",  # Not enough data to make split
        "Study Type",  # Not enough data to make split
    ]
    for task in tasks:

        task_lower = task.replace(' ', '_').lower() + '.csv'
        file = find_file_in_dir(task_lower, data_path)

        if 'onehot' in file:
            data_handler = PsyNamicMultiLabel(file)
        else:
            data_handler = PsyNamicSingleLabel(file, task)
        try:
            print(f'Processing {task}')
            data_handler.print_label_dist()
            data_handler.get_strat_split(use_val=True)
            data_handler.save_split(
                f'{data_path}/{task.replace(" ", "_").lower()}/')
            print('\n')
        except ValueError as e:
            print(e)
            # TODO: Handle to small splits
            print(f'Could not split {task}')

    file = 'data/raw_data/asreview_dataset_all_Psychedelic Study.csv'
    data_handler = PsychNamicRelevant(
        file, 'record_id', 'title', 'abstract', 'included')
    data_handler.get_strat_split(use_val=True)
    data_handler.save_split(f'data/prepared_data/asreview_dataset_all/')


def prepare_all(outfile: str):
    file = 'data/raw_data/asreview_dataset_all_Psychedelic Study.csv'
    df = pd.read_csv(file)
    # get distribution of included column, including nan
    print(df['included'].value_counts(dropna=False))
    print(len(df))
    data_handler = PsychNamicRelevant(
        file, 'record_id', 'title', 'abstract', 'included')
    # where included is 1
    df = data_handler.df[data_handler.df['labels'] == 1]
    print(len(df))
    df = df.drop(columns=['labels'])
    df.to_csv(outfile, index=False)


if __name__ == '__main__':
    # Training 1
    list_jsonl = [
        "data/prodigy_exports/prodigy_export_ben_95_20240423_113434.jsonl",
        "data/iaa/iaa_round1_50/iaa_resolution/prodigy_export_review_all_text_50_20240418_20240607_145354.jsonl",
        "data/prodigy_exports/prodigy_export_ben_24_20240425_152801.jsonl",
        "data/iaa/iaa_round2_40/iaa_resolution/prodigy_export_review_all_text_40_20240523_20240705_183405.jsonl",
        "data/prodigy_exports/prodigy_export_pia_250_20240423_113437_20240720_135743.jsonl"
    ]
    annotators = [
        'Ben',
        'IAA Resolution',
        'Ben',
        'IAA Resolution',
        'Pia'
    ]

    # Fix for duplicates --> put Ben's files first
    list_json_bio = [
        'data/prodigy_exports/prodigy_export_ben_95_20240423_113434.jsonl',
        # Data appearing 3 times
        'data/prodigy_exports/prodigy_export_ben_24_20240425_152801.jsonl',
        'data/iaa/iaa_round1_50/iaa_resolution/prodigy_export_review_all_token_50_20240418_20240607_145359.jsonl',
        # Data appearing 3 times
        'data/iaa/iaa_round2_40/iaa_resolution/prodigy_export_review_all_token_40_20240523_20240705_183410.jsonl',
        'data/prodigy_exports/prodigy_export_pia_250_20240423_113437_20240720_135743.jsonl'
    ]
    # prodigy_data = ProdigyDataCollector(
    #    list_jsonl, annotators, expert_annotator='Ben')
    # prepare_train_data(list_jsonl, annotators, outpath='data/prepared_data/training_round1')
    # prepare_splits('data/prepared_data/training_round1')
    # prepare_bio_data(list_json_bio, 'record_id',
    #                  'data/prepared_data/ner_bio.jsonl')
    # outfile = 'data/prepared_data/all/psychedelic_study_relevant.csv'
    # prepare_all(outfile)

    # Second round training
    list_jsonl = [
        'data/iaa/iaa_round1_50/iaa_resolution/prodigy_export_review_all_text_50_20240418_20240607_145354.jsonl',
        'data/iaa/iaa_round1_50/iaa_resolution/prodigy_export_review_all_token_50_20240418_20240607_145359.jsonl',
        'data/iaa/iaa_round2_40/iaa_resolution/prodigy_export_review_all_text_40_20240523_20240705_183405.jsonl',
        'data/iaa/iaa_round2_40/iaa_resolution/prodigy_export_review_all_token_40_20240523_20240705_183410.jsonl',
        'data/prodigy_exports/prodigy_export_ben_95_20240423_113434.jsonl',
        'data/prodigy_exports/prodigy_export_ben_24_20240425_152801_reordered.jsonl',
        'data/prodigy_exports/prodigy_export_pia_250_20240730_095458_20240812_192652.jsonl',
        'data/prodigy_exports/prodigy_export_ben_582_double_annot_review_text_20240812_20241129_105310.jsonl',
        'data/prodigy_exports/prodigy_export_ben_582_double_annot_review_token_20240812_20241203_193705_token_corrected.jsonl'
    ]

    annotators = [
        'IAA Resolution',
        'IAA Resolution',
        'IAA Resolution',
        'IAA Resolution',
        'Ben',
        'Ben',
        'Pia',
        'Ben_double_annot',
        'Ben_double_annot'
    ]

    purposes = [
        'class',
        'ner',
        'class',
        'ner',
        'both',
        'both',
        'both',
        'class',
        'ner',]
    prodigy_data = ProdigyDataCollector(
        list_jsonl, annotators, expert_annotator='Ben_double_annot', purposes=purposes)
    
    prepare_train_data(
        prodigy_data, outpath='data/prepared_data/training_round2')
    prepare_splits('data/prepared_data/training_round2')
    prepare_bio_data(list_jsonl, 'record_id', 'data/prepared_data/training_round2/ner.jsonl',
                     purposes, expert_annot='Ben_double_annot')
