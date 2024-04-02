# reference code https://github.com/explosion/prodigy-recipes/tree/master/tutorials/span-and-textcat
import spacy
import prodigy
import json
from prodigy.components.preprocess import add_tokens
from prodigy.components.loaders import JSONL

# start Prodigy server
# python -m prodigy span-and-textcat pubmed_psych en ./input/psychdelic_study_50_20240312.jsonl -F ./recipe.py


@prodigy.recipe(
    "span-and-textcat",
    dataset=("Dataset to save annotations into", "positional", None, str),
    lang=("Language to use", "positional", None, str),
    file_in=("Path to examples.jsonl file", "positional", None, str),
)

def custom_recipe(dataset, lang, file_in):
    span_labels = ["Application"]
    with open('labels.json', 'r') as infile:
        labels =  json.load(infile)
    
    # concat all value lists
    textcat_labels =  [label for sublist in labels.values() for label in sublist]
    
    def add_options(stream):
        for ex in stream:
            ex['options'] = [
                {"id": i, "text": lab} for i, lab in enumerate(textcat_labels)
            ]
            yield ex
                  
    def get_keymap(labels: dict[list]) -> dict[str, str]:
        keymap = {}
        i = 0
        for key, values in labels.items():
            for value in values:
                keymap[str(i)] = key
                i += 1
        return keymap
    
    nlp = spacy.blank(lang)
    stream = JSONL(file_in)
    stream = add_tokens(nlp, stream)
    stream = add_options(stream)

    blocks = [
        {"view_id": "spans_manual"},
        {"view_id": "choice", "text": None},
    ]
    return {
        "view_id": "blocks",  # Annotation interface to use
        "dataset": dataset,  # Name of dataset to save annotations
        "stream": stream,  # Incoming stream of examples
        "config": {  # Additional config settings, mostly for app UI
            "lang": nlp.lang,
            "labels": span_labels,
            "blocks": blocks,
            "keymap_by_label": get_keymap(labels),
            "choice_style": "multiple"
        },
    }