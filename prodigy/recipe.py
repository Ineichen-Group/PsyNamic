# reference code https://github.com/explosion/prodigy-recipes/tree/master/tutorials/span-and-textcat
import spacy
import prodigy 
from prodigy.components.preprocess import add_tokens
from prodigy.components.loaders import JSONL


@prodigy.recipe(
    "span-and-textcat",
    dataset=("Dataset to save annotations into", "positional", None, str),
    lang=("Language to use", "positional", None, str),
    file_in=("Path to examples.jsonl file", "positional", None, str)
)
def custom_recipe(dataset, lang, file_in):
    span_labels = ["Study type", "Substance", "Application", "Number of participants", "Sex of participants", "Clinical trial phase", "Adverse event(s)", "Conclusion"]
    textcat_labels = ["clinical study", "not clinical study"]

    def add_options(stream):
        for ex in stream:
            ex['options'] = [
                {"id": lab, "text": lab} for lab in textcat_labels
            ]
            yield ex

    nlp = spacy.blank(lang)
    stream = JSONL(file_in)
    # stream = add_tokens(nlp, stream, use_chars=None)
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
            "keymap_by_label": {
                "0": "species", 
                "1": "species", 
                "2": "species", 
                "3": "species", 
                "4": "species", 
                "5": "species", 
                "6": "species", 
                "7": "species",
                "8": "species",
                "9": "outcome", 
                "10": "outcome", 
                "11": "outcome", 
                "12": "outcome", 
                "13": "outcome", 
                "14": "control" 
                
                
            },
            "choice_style": "multiple"
        },
    }
