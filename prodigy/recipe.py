# reference code https://github.com/explosion/prodigy-recipes/tree/master/tutorials/span-and-textcat
import spacy
import prodigy
import json
import seaborn as sns
from typing import Optional
from prodigy import recipe, log, get_stream
from prodigy.components.preprocess import add_tokens
from prodigy.models.matcher import PatternMatcher
from prodigy.components.loaders import JSONL
from prodigy.types import RecipeSettingsType
import copy


# start Prodigy server
# python -m prodigy span-and-textcat pubmed_psych en ./input/psychedelic_study_50_20240312.jsonl -F ./recipe.py


@prodigy.recipe(
    "span-and-textcat",
    dataset=("Dataset to save annotations into", "positional", None, str),
    lang=("Language to use", "positional", None, str),
    file_in=("Path to examples.jsonl file", "positional", None, str),
    patterns=("Path to match patterns file", "option", "pt", str),

)
def custom_recipe(
    dataset: str,
    lang: str,
    file_in: str,
    patterns: Optional[str] = None
) -> RecipeSettingsType:

    # Fetch labels from json files
    with open('span_labels.json', 'r') as infile:
        span_labels = json.load(infile)

    with open('choice_labels.json', 'r') as infile:
        labels = json.load(infile)

    colors = get_colors(labels)
    keymap = get_keymap(labels)
    # Flatten the labels
    flat_labels = [f'{group}: {label}' for group,
                   sublist in labels.items() for label in sublist]

    nlp = spacy.load(f'blank:{lang}')
    stream = get_stream(file_in)
    stream = add_options_and_highlights(stream, flat_labels, colors, nlp, patterns)
    stream = add_tokens(nlp, stream)
    
    blocks = [
        {"view_id": "html", "html_template": '<p>doi: <a href="{{pubmed_url}}" target="_blank">{{doi}}</a></p><p>Published in: {{secondary_title}}</p>'},
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
            "keymap_by_label": keymap,
            "choice_style": "multiple"
        },
    }

def add_options_and_highlights(
        stream,
        flat_labels: list[str],
        colors: list[str],
        nlp: spacy.language.Language,
        patterns: str):
    
    matcher = PatternMatcher(
        nlp,
        label_span=False, # no label added to the span
        label_task=False, # no label on top level task
        combine_matches=True, # show all matches in one task
        all_examples=True, # alle examples are returned
    )

    matcher = matcher.from_disk(patterns)

    options = [
        {"id": i, "text": lab, "style": {"background-color": c}} for i, (lab, c) in enumerate(zip(flat_labels, colors))
    ]
    for score, eg in matcher(stream):
        task = copy.deepcopy(eg)
        task["options"] = options
        yield task

def convert_color(value: float):
    """Convert a value between 0 and 1 to a color value between 0 and 255."""
    return round(value * 255)

def get_colors(labels: dict[str, list[str]]):
    """Get a list of colors as a rbga strings, one for each label.
    Can be fed directly into css"""
    color_palette = sns.color_palette("Paired", len(labels))
    color_strings = [f'rgba({convert_color(r)}, {convert_color(g)}, {convert_color(b)}, 0.3)' for r, g, b in color_palette]
    colors = [color for sublist, color in zip(
        labels.values(), color_strings) for label in sublist]
    return colors

# Hacky solution to use keymapping for grouping the labels
def get_keymap(labels: dict[list]) -> dict[str, str]:
    """ Use keymap to group the labels."""
    keymap = {}
    i = 0
    for key, values in labels.items():
        for value in values:
            keymap[str(i)] = key
            i += 1
    return keymap
