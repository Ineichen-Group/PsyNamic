import json
from collections import OrderedDict
import os
import random

synonyms = {
    'substance: ketamine': 'substances: ketamine',
    'substance: ibogaine': 'substances: ibogaine',
    'substance: mdma': 'substances: mdma',
    'substance: ayahuasca': 'substances: ayahuasca',
    'substance: dmt': 'substances: dmt',
    'substance: combination therapy': 'substance: combination therapy',
    'substance: lsd': 'substances: lsd',
    'substance: 5-meo-dmt': 'substances: 5-meo-dmt',
    'substance: psilocybin': 'substances: psilocybin',
    'substance: combination therapy': 'substances: combination therapy',
    'conclusion: negative': 'study conclusion: negative',
    'conclusion: positive': 'study conclusion: positive',
    'conclusion: not applicable': 'study conclusion: not applicable',
    'conclusion: mixed': 'study conclusion: mixed',
    'condition: pain': 'condition: (chronic) pain',
}

unknown = set()

files_to_double_annotate = [
    'data/prodigy_exports/prodigy_export_ben_100_20240411_20240416_162821.jsonl',
    'data/prodigy_exports/prodigy_export_ben_100_20240418_20240423_075953.jsonl',
    'data/prodigy_exports/prodigy_export_julia_250_20240423_113435_20240812_012727.jsonl',
    'data/prodigy_exports/prodigy_export_bernard_250_20240423_113436_20240713_171907.jsonl',
    'data/prodigy_exports/prodigy_export_pia_250_20240423_113437_20240720_135743.jsonl',
]

styling_helper_file = 'data/prodigy_exports/prodigy_export_julia_250_20240423_113435_20240812_012727.jsonl'
annotation_tasks = 'prodigy/choice_labels.json'
outfile = 'data/prodigy_exports/prodigy_export_double_annotated_20240812.jsonl'

# if already exists, delete it
try:
    os.remove(outfile)
except FileNotFoundError:
    pass


def get_id2label_from_jsonl(jsonl_file: str) -> tuple[OrderedDict, OrderedDict]:
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        # read in the first line
        line = json.loads(f.readline(), object_pairs_hook=OrderedDict)
        options = line['options']
        id2label = OrderedDict()
        for opt in options:
            id2label[opt['id']] = opt['text'].lower()
    return id2label, options


# Read in the annotation tasks
with open(annotation_tasks, 'r', encoding='utf-8') as f:
    # load annotation task in an ordered dict
    annotation_task_groups = json.load(f, object_pairs_hook=OrderedDict)
    cur_task2id = {}
    idx = 0
    for group in annotation_task_groups.values():
        for group, opts in group.items():
            for opt in opts:
                label = f'{group}: {opt}'.lower()
                cur_task2id[label] = idx
                idx += 1
    cur_id2task = {v: k for k, v in cur_task2id.items()}

new_id2taskstyle, new_options = get_id2label_from_jsonl(styling_helper_file)

# Check if the cur_id2task is consistent with the ref_id2taskstyle
for k, v in cur_id2task.items():
    if v != new_id2taskstyle[k]:
        print(f"Warning: {v} != {new_id2taskstyle[k]}")

all_data = []

with open(outfile, 'w', encoding='utf-8') as out:
    for file in files_to_double_annotate:
        id2task, _ = get_id2label_from_jsonl(file)
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line, object_pairs_hook=OrderedDict)
                line['options'] = new_options
                accepted = line['accept']
                new_accepted = []
                # replace selected
                for act_id in accepted:
                    task = id2task[act_id]
                    try:
                        new_accepted.append(cur_task2id[task])
                    except KeyError:
                        try:
                            new_accepted.append(cur_task2id[synonyms[task]])
                        except KeyError:
                            unknown.add(task)

                line['accept'] = new_accepted
                all_data.append(line)
    # write to file in random order:
    random.shuffle(all_data)
    for line in all_data:
        out.write(json.dumps(line, ensure_ascii=False) + '\n')

print(unknown)
