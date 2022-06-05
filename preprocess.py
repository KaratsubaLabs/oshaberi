
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


def bag_words():
    raise NotImplementedError
