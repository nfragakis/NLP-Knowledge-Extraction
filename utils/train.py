import pandas as pd
import numpy as np
import re

from tqdm import tqdm
tqdm.pandas()

import spacy
assert spacy.__version__ == '2.1.0'
from spacy.matcher import Matcher
from spacy.util import minibatch, compounding
from spacy.util import decaying
import random


def custom_optimizer(optimizer, learn_rate=0.0001, beta1=0.9, beta2=0.999, eps=1e-8, L2=1e-6, max_grad_norm=1.0):
    """
    Function to customizer spaCy default optimizer
    """
    optimizer.learn_rate = learn_rate
    optimizer.beta1 = beta1
    optimizer.beta2 = beta2
    optimizer.eps = eps
    optimizer.L2 = L2
    optimizer.max_grad_norm = max_grad_norm

    return optimizer


def train_ner(TRAINING_DATA, EPOCHS, lr, log_every=100):
    optimizer = nlp.entity.create_optimizer()

    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

    # TRAINING THE MODEL
    with nlp.disable_pipes(*unaffected_pipes):

        optimizer = nlp.resume_training(component_cfg={"ner": {"conv_window": 3, "self_attn_depth": 2}})
        optimizer = custom_optimizer(optimizer, lr)

        dropout = decaying(0.6, 0.2, 1e-4)

        for i in range(EPOCHS):
            # shuufling examples  before every iteration
            random.shuffle(TRAINING_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAINING_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,
                    annotations,
                    drop=next(dropout),
                    sgd=optimizer,
                    losses=losses,
                )
            if (i + 1) % log_every == 0: print(f"Iteration {i + 1} ==> Loss: ", losses['ner'])


def build_ner_training(txt, pattern, coref=False):
    matcher = Matcher(nlp.vocab)

    for pattern in patterns:
        # set singular and plural patterns as well as base part catcher

        pattern1 = [{'DEP': 'ROOT', 'OP': '?'},
                    {'DEP': 'meta', 'OP': '?'},
                    {'DEP': 'compound', 'OP': '?'},
                    {'DEP': 'nsubj', 'OP': '?'},
                    {'LOWER': pattern}]

        pattern2 = [{'DEP': 'ROOT', 'OP': '?'},
                    {'DEP': 'meta', 'OP': '?'},
                    {'DEP': 'compound', 'OP': '?'},
                    {'DEP': 'nsubj', 'OP': '?'},
                    {'LOWER': pattern + 's'}]

        # general regex to capture prt nbr
        pattern3 = [{'DEP': 'ROOT', 'OP': '?'},
                    {'DEP': 'meta', 'OP': '?'},
                    {'DEP': 'compound', 'OP': '?'},
                    {'DEP': 'nsubj', 'OP': '?'},
                    {'TEXT': {'REGEX': '(\w*\d[\w\d]+)'}}]

        if len(pattern.split(' ')) == 2:
            # drop last element
            pattern1.pop()
            pattern2.pop()

            # add new multi word pattern
            pattern1.append({'LOWER': pattern.split(' ')[0]})
            pattern1.append({'LOWER': pattern.split(' ')[1]})
            pattern2.append({'LOWER': pattern.split(' ')[0]})
            pattern2.append({'LOWER': pattern.split(' ')[1] + 's'})

        matcher.add("PRODUCT", None, pattern1, pattern2, pattern3)

    TRAINING_DATA = []

    for i, doc in enumerate(nlp.pipe(txt)):

        # Resolve coreferences
        if coref:
            doc = nlp(doc._.coref_resolved)

        # Match on the doc and create a list of matched spans
        spans = [doc[start:end] for match_id, start, end in matcher(doc)]

        # Get (start character, end character, label) tuples of matches
        entities = [(span.start_char, span.end_char, "PRODUCT") for span in spans]

        # Format the matches as a (doc.text, entities) tuple
        training_example = (doc.text, {"entities": entities})

        # Append the example to the training data
        TRAINING_DATA.append(training_example)

    print(f'{len(TRAINING_DATA)} Examples Captured')

    return TRAINING_DATA