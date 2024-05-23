import numpy as np
import itertools
import pandas as pd
from nltk.metrics.agreement import AnnotationTask


# TODO: Add confidence intervals via bootstrapping, s. 
def calculate_krippendorff_alpha(nltk_formatted_input: list[tuple[str, str, frozenset]]) -> float:
    """
    nltk_formatted_input needs to have the structure: [(coder, item, label)]
    s. https://www.nltk.org/api/nltk.metrics.agreement.html
    e.g.
    data = [
    ('c1', '1', frozenset({'v1', 'v2'})), ('c2', '1', frozenset({'v1'})),
    ('c1', '2', frozenset({'v2'})), ('c2', '2', frozenset({'v2', 'v3'})),
    ('c1', '3', frozenset()), ('c2', '3', frozenset({'v1'})),
    ('c1', '4', frozenset({'v3'})), ('c2', '4', frozenset({'v3', 'v2'}))
    ]

    Args:
        nltk_formatted_input (list[tuple[str, str, frozenset]]): _description_
        annotators (list): _description_
    """
    task = AnnotationTask(data=nltk_formatted_input)
    return task.alpha()

def main():
   data = data = [
    ('c1', '1', frozenset({'v1', 'v2'})), ('c2', '1', frozenset({'v1', 'v2'})),
    ('c3', '1', frozenset({'v1', 'v2'})), ('c4', '1', frozenset({'v1', 'v2'})),
    
    ('c1', '2', frozenset({'v2'})), ('c2', '2', frozenset({'v2'})),
    ('c3', '2', frozenset({'v2'})), ('c4', '2', frozenset({'v2'})),
    
    ('c1', '3', frozenset({'v3'})), ('c2', '3', frozenset({'v3'})),
    ('c3', '3', frozenset({'v3'})), ('c4', '3', frozenset({'v3'})),
    
    ('c1', '4', frozenset({'v1'})), ('c2', '4', frozenset({'v1', 'v2'})),
    ('c3', '4', frozenset({'v1'})), ('c4', '4', frozenset({'v1'})),
    
    ('c1', '5', frozenset({'v2', 'v3'})), ('c2', '5', frozenset({'v2', 'v3'})),
    ('c3', '5', frozenset({'v2', 'v3'})), ('c4', '5', frozenset({'v2', 'v3'}))
]

   print(calculate_krippendorff_alpha(data))


if __name__ == '__main__':

    main()
