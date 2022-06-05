
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

STOP_WORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()


def tokenize(sentence):
    return word_tokenize(sentence)


def filter_stopwords(tokenized):
    return [token for token in tokenized if token.casefold() not in STOP_WORDS]


def stem(tokenized):
    return [stemmer.stem(token) for token in tokenized]


def bag_words(tokenized, word_dict):
    bag = np.zeros(len(word_dict), dtype=np.float32)
    for i, token in enumerate(tokenized):
        if token in word_dict:
            bag[i] = 1.0

    return bag

