# reference code https://github.com/explosion/prodigy-recipes/tree/master/tutorials/span-and-textcat
import copy
import json
import re
from typing import Optional

import seaborn as sns
import spacy

import prodigy
from prodigy import set_hashes, log
from prodigy.components.preprocess import add_tokens
from prodigy.components.stream import get_stream
from prodigy.models.matcher import PatternMatcher
from prodigy.types import RecipeSettingsType

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

    patterns = create_patterns_jsonl(patterns)
    # Fetch labels from json files
    with open('span_labels.json', 'r') as infile:
        span_labels = json.load(infile)

    with open('choice_labels.json', 'r') as infile:
        labels = json.load(infile)

    flat_labels = []
    for _, sub_labels in labels.items():
        for label, values in sub_labels.items():
            for value in values:
                flat_labels.append(f"{label}: {value}")

    colors = get_colors(labels)

    nlp = spacy.load(f'blank:{lang}')
    stream = get_stream(file_in, dedup=False, rehash=True)
    stream = (set_hashes(eg, task_keys=("annotation",), overwrite=True)
              for eg in stream)
    stream = add_options_and_highlights(
        stream, flat_labels, colors, nlp, patterns, labels)
    stream = add_tokens(nlp, stream)

    blocks = [
        {"view_id": "html", "html_template": '<p>Annotating: <b>{{annotation}}</b></p><p>doi: <a href="{{pubmed_url}}" target="_blank">{{doi}}</a></p><p>Published in: {{secondary_title}}</p>'},
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
            "keymap_by_label": None,
            "choice_style": "multiple"
        },
    }


def add_options_and_highlights(
        stream,
        flat_labels: list[str],
        colors: list[str],
        nlp: spacy.language.Language,
        patterns: str,
        labels: dict[str, list[str]] = None):

    matcher = PatternMatcher(
        nlp,
        label_span=False,  # no label added to the span
        label_task=False,  # no label on top level task
        combine_matches=True,  # show all matches in one task
        all_examples=True,  # alle examples are returned
        task_hash_keys=("annotation",),
    )

    matcher = matcher.from_disk(patterns)

    options = [
        {"text": lab, "style": {"background-color": c}} for i, (lab, c) in enumerate(zip(flat_labels, colors))
    ]
    # for score, eg in matcher(stream):
    for eg in stream:
        current_annotation = eg['annotation']
        filtered_options = filter_options(options, labels[current_annotation])
        task = copy.deepcopy(eg)
        task["options"] = filtered_options
        yield task


def filter_options(options: list[dict], label_group: dict):
    """Filter the options based on the labels_sorted dictionary."""
    filtered_options = []
    flattened_labels = [label for label in label_group.keys()]
    for option_dict in options:
        label_group = option_dict['text'].split(': ')[0]
        if label_group in flattened_labels:
            filtered_options.append(option_dict)
    # add ids to the options
    for i, option in enumerate(filtered_options):
        option['id'] = i

    return filtered_options


def convert_color(value: float):
    """Convert a value between 0 and 1 to a color value between 0 and 255."""
    return round(value * 255)


def get_colors(labels: dict) -> list[str]:
    """Get a list of colors as a rbga strings, one for each label.
    Can be fed directly into css"""
    labels_flatter = {}

    for _, label_group in labels.items():
        for label, options in label_group.items():
            labels_flatter[label] = options

    color_palette = sns.color_palette("Paired", len(labels_flatter))
    color_strings = [
        f'rgba({convert_color(r)}, {convert_color(g)}, {convert_color(b)}, 0.3)' for r, g, b in color_palette]
    colors = [color for sublist, color in zip(
        labels_flatter.values(), color_strings) for label in sublist]

    return colors


def create_patterns_jsonl(input_file: str) -> str:
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    patterns = []

    for line in lines:
        line = line.strip()

        if line.startswith('*'):
            pattern = {"lower": {"REGEX": f'.{line}.*'}}
        else:
            pattern = {"lower": line}
        pattern = {"pattern": [pattern], "label": "P-Highlight"}

        patterns.append(pattern)
    output_file = input_file.replace('.txt', '.jsonl')

    return output_file
