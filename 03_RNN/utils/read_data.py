from io import open
import unicodedata
import string
import re
import random
import torch
from .format_data import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def filterPair(p,MAX_LENGTH=10):

    c1 = len(p[0].split(' ')) < MAX_LENGTH
    c2 = len(p[1].split(' ')) < MAX_LENGTH
    eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
    )

    c3 = p[1].startswith(eng_prefixes)
    return  c1 and c2 and c3 


def filterPairs(pairs,MAX_LENGTH=10):
    return [pair for pair in pairs if filterPair(pair,MAX_LENGTH=10)]


def prepareData(lang1, lang2, reverse=False, MAX_LENGTH=10):
    MAX_LENGTH=MAX_LENGTH

    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))

    pairs = filterPairs(pairs,MAX_LENGTH)

    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    
    SOS_token = 0
    EOS_token = 1
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair,input_lang,output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)