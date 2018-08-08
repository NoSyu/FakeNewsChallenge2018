import nltk
import re
from sklearn import feature_extraction

_wnl = nltk.WordNetLemmatizer()
def normalize_word(w):
    return _wnl.lemmatize(w).lower()


def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]


def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric

    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()


def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]

def data_prepro(text):
    text = clean(text)
    text = get_tokenized_lemmas(text)
    text = remove_stopwords(text)
    text = ' '.join(text)
    return text
