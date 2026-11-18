import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


import re

# Hyperparameters of the transformation
SYNONYM_PROB = 0.30  # prob. that an eligible word is replaced by a WordNet synonym
TYPO_PROB = 0.20     # prob. that a (non-replaced) word gets a single keyboard typo
MIN_WORD_LEN = 4     # only words with at least this many letters are touched

# Adjacent keys on a QWERTY keyboard
QWERTY_NEIGHBORS = {
    'q': 'wa', 'w': 'qeas', 'e': 'wrsd', 'r': 'etdf', 't': 'ryfg', 'y': 'tugh', 'u': 'yihj',
    'i': 'uojk', 'o': 'ipkl', 'p': 'ol', 'a': 'qwsz', 's': 'awedxz', 'd': 'serfcx',
    'f': 'drtgvc', 'g': 'ftyhbv', 'h': 'gyujnb', 'j': 'huikmn', 'k': 'jiolm', 'l': 'kop',
    'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk',
}

# Words we never modify: function words and negations/intensifiers whose change could flip
# or distort the sentiment (e.g. WordNet maps "not" -> "non"), plus the HTML "br" tag in IMDB.
PROTECTED = {
    'not', 'never', 'none', 'nothing', 'nobody', 'neither', 'nor', 'without', 'cannot',
    'this', 'that', 'these', 'those', 'there', 'their', 'they', 'them', 'then', 'than',
    'what', 'when', 'where', 'which', 'while', 'with', 'from', 'have', 'been', 'were',
    'will', 'would', 'could', 'should', 'about', 'into', 'just', 'also', 'very', 'some',
    'more', 'most', 'only', 'even', 'such', 'your', 'here', 'does', 'doesn', 'didn',
    'isn', 'wasn', 'aren', 'weren', 'don', 'won', 'br',
}


def _match_case(src, tgt):
    if src.isupper():
        return tgt.upper()
    if src[0].isupper():
        return tgt[0].upper() + tgt[1:]
    return tgt


def _synonym(word):
    lower = word.lower()
    candidates = set()
    for syn in wordnet.synsets(lower):
        for lemma in syn.lemmas():
            name = lemma.name().replace('_', ' ')
            if name.lower() != lower and name.isascii():
                candidates.add(name)
    if not candidates:
        return None
    return _match_case(word, random.choice(sorted(candidates)))


def _typo(word):
    positions = [i for i, ch in enumerate(word) if ch.lower() in QWERTY_NEIGHBORS]
    if not positions:
        return word
    i = random.choice(positions)
    new_ch = random.choice(QWERTY_NEIGHBORS[word[i].lower()])
    if word[i].isupper():
        new_ch = new_ch.upper()
    return word[:i] + new_ch + word[i + 1:]


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Transformation: "careless paraphrasing user". Each eligible word (alphabetic, >= 4 letters,
    # not in PROTECTED) is independently
    #   (1) with prob SYNONYM_PROB replaced by a random WordNet synonym (any synset/lemma), else
    #   (2) with prob TYPO_PROB given one typo: a random letter swapped for an adjacent QWERTY key.
    # Splitting on non-letter runs (and keeping them) preserves punctuation, spacing and HTML tags.

    pieces = re.split(r'([^A-Za-z]+)', example["text"])
    out = []
    for piece in pieces:
        if (len(piece) >= MIN_WORD_LEN and piece.isalpha() and piece.lower() not in PROTECTED):
            r = random.random()
            if r < SYNONYM_PROB:
                syn = _synonym(piece)
                if syn is not None:
                    piece = syn
            elif r < SYNONYM_PROB + TYPO_PROB:
                piece = _typo(piece)
        out.append(piece)
    example["text"] = "".join(out)

    ##### YOUR CODE ENDS HERE ######

    return example
