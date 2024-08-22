from data.prodigy_data_reader import ProdigyDataCollector
from datahandler import PsyNamicSingleLabel, PsychNamicRelevant, PsyNamicMultiLabel, DataHandlerBIO
import os
import time
import json
import pandas as pd
from stride_utils.prodigy import ner_process_file_and_save_to_bio_format


def prepare_train_data(list_jsonl: list[str], annotators: list[str]) -> None:
    prodigy_data = ProdigyDataCollector(list_jsonl, annotators)
    tasks = prodigy_data.tasks.keys()
    prepared_data = 'data/prepared_data/'
    for task in tasks:
        task_string = task.replace(' ', '_').lower()
        date = time.strftime("%Y%m%d")
        meta_data = {
            "Date": date,
            "Task": task,
            "Files": list_jsonl,
            "Size": len(prodigy_data),
            "Is_multilabel": prodigy_data.is_multilabel(task)
        }
        if prodigy_data.is_multilabel(task):
            meta_file = os.path.join(
                prepared_data, f'onehot_{task_string}_meta.json')
            task_df = prodigy_data.get_onehot_task_df(task)
            task_df.to_csv(os.path.join(
                prepared_data, f'onehot_{task_string}.csv'), index=False)
            with open(meta_file, 'w') as f:
                json.dump(meta_data, f, indent=4, ensure_ascii=False)

        else:
            meta_file = os.path.join(
                prepared_data, f'{task_string}_meta.json')
            label_to_int, task_df = prodigy_data.get_label_task_df(task)
            task_df.to_csv(os.path.join(
                prepared_data, f'{task_string}.csv'), index=False)

            meta_data["Int_to_label"] = {v: k for k, v in label_to_int.items()}
            with open(meta_file, 'w') as f:
                json.dump(meta_data, f, indent=4, ensure_ascii=False)


def find_file_in_dir(file_string: str, dir: str) -> str:
    for file in os.listdir(dir):
        if file_string in file:
            return os.path.join(dir, file)
    return None


def prepare_bio_data(list_jsonl: list[str], id_field: str, outfile: str) -> str:
    outfiles = []
    outfile_path = os.path.dirname(outfile)
    for file in list_jsonl:
        output_name = os.path.join(
            outfile_path, f'{os.path.basename(file).replace(".jsonl", "_bio")}')
        output_file = ner_process_file_and_save_to_bio_format(
            file, output_name, id_field)
        outfiles.append(output_file)
    # Delete outfile if it exists
    if os.path.exists(outfile):
        os.remove(outfile)

    line_count = 0
    ids = set()

    # Merge all files
    with open(outfile, 'a') as out:
        for file in outfiles:
            with open(file, 'r') as f:
                for line in f:
                    # parse line
                    parsed_line = json.loads(line)
                    # check if id is unique
                    id = parsed_line['id']
                    if id in ids:
                        print(
                            f'Skipping duplicate id {parsed_line["id"]} in {file}')
                    else:
                        ids.add(id)
                        out.write(line)
                        line_count += 1
    # append line count to file name
    new_file_name = outfile.replace('.jsonl', f'_{line_count}.jsonl')
    os.rename(outfile, new_file_name)

    # Remove temp files
    for file in outfiles:
        os.remove(file)

    bio_datahandler = DataHandlerBIO(new_file_name)
    print(bio_datahandler.label2id)
    bio_datahandler.get_split(use_val=True)
    bio_datahandler.save_split('data/prepared_data/ner_bio/')


def prepare_splits():

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
        # "Study Conclusion", # Not enough data to make split
        # "Study Type", # Not enough data to make split
    ]
    data_path = 'data/prepared_data/'
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
                f'data/prepared_data/{task.replace(" ", "_").lower()}/')
            print('\n')
        except ValueError:
            breakpoint()
            # TODO: Handle to small splits
            print(f'Could not split {task}')

    file = 'data/raw_data/asreview_dataset_all_Psychedelic Study.csv'
    data_handler = PsychNamicRelevant(
        file, 'record_id', 'title', 'abstract', 'included')
    data_handler.get_strat_split(use_val=True)
    data_handler.save_split(f'data/prepared_data/asreview_dataset_all/')


# if main equal name
if __name__ == '__main__':
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
        'data/prodigy_exports/prodigy_export_ben_24_20240425_152801.jsonl', # Data appearing 3 times
        'data/iaa/iaa_round1_50/iaa_resolution/prodigy_export_review_all_token_50_20240418_20240607_145359.jsonl',
        'data/iaa/iaa_round2_40/iaa_resolution/prodigy_export_review_all_token_40_20240523_20240705_183410.jsonl',# Data appearing 3 times
        'data/prodigy_exports/prodigy_export_pia_250_20240423_113437_20240720_135743.jsonl'
    ]

    # prepare_train_data(list_jsonl, annotators)
    # prepare_splits()
    prepare_bio_data(list_json_bio, 'record_id',
                     'data/prepared_data/ner_bio.jsonl')
