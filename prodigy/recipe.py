# reference code https://github.com/explosion/prodigy-recipes/tree/master/tutorials/span-and-textcat
import spacy
import prodigy
import json
import seaborn as sns
from prodigy.components.preprocess import add_tokens
from prodigy.components.loaders import JSONL
from prodigy.types import RecipeSettingsType

# start Prodigy server
# python -m prodigy span-and-textcat pubmed_psych en ./input/psychdelic_study_50_20240312.jsonl -F ./recipe.py


@prodigy.recipe(
    "span-and-textcat",
    dataset=("Dataset to save annotations into", "positional", None, str),
    lang=("Language to use", "positional", None, str),
    file_in=("Path to examples.jsonl file", "positional", None, str),
)

def custom_recipe(dataset:str, lang: str, file_in: str) -> RecipeSettingsType:
    with open('span_labels.json', 'r') as infile:
        span_labels =  json.load(infile)
        
    with open('choice_labels.json', 'r') as infile:
        labels =  json.load(infile)
    
    # Flatten the labels
    textcat_labels =  [f'{group}: {label}' for group, sublist in labels.items() for label in sublist]
    
    # Get color palette for each label group
    def convert_color(value: float):
        """Convert a value between 0 and 1 to a color value between 0 and 255."""
        return round(value * 255)
    
    color_palette = sns.color_palette("Paired", len(labels))
    color_strings = [f'rgba({convert_color(r)}, {convert_color(g)}, {convert_color(b)}, 0.3)' for r, g, b in color_palette]
    colors = [color for sublist, color in zip(labels.values(), color_strings) for label in sublist]

    def add_options(stream):
        for ex in stream:
            ex['options'] = [
                {"id": i, "text": lab, "style": {"background-color": c}} for i, (lab, c) in enumerate(zip(textcat_labels, colors))
            ]
            yield ex
      
    # Hacky solution to use keymapping for grouping the labels        
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
        {"view_id": "html", "html_template": '<p><b>{{title}}</b></p><p>doi: <a href="{{pubmed_url}}" target="_blank">{{doi}}</a></p><p>Published in: {{secondary_title}}</p>'},
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