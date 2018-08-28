from keras_model.feature_helper import word_ngrams
from os import path
import os
import pickle
from nltk.corpus import stopwords
import nltk
import string
from sklearn import feature_extraction
from keras_model.utils.data_helpers import normalize_word
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from keras_model.utils.dataset import DataSet
# from keras_model.utils.generate_test_splits import kfold_split
import numpy as np

# def tfidf_100(head, body, path, filename):

def word_ngrams_concat(headlines, bodies, max_features=200, ngram_range=(1, 1),
                       use_idf=False, norm='l2', lemmatize=False,
                       term_freq=True, include_holdout=False):
    """
    Takes parameters to fit a TfidfVectorizer on the training and test stance (optional holdout)
    and transforms the headlines and bodies seperately on it. At the end, both feature vectors
    get concatenated and returned
    """
    def get_features(vocab):
        if  term_freq == True:
            vectorizer_head = TfidfVectorizer(vocabulary=vocab, use_idf=use_idf,
                                             norm=norm, stop_words='english')
        else:
            vectorizer_head = CountVectorizer(vocabulary=vocab,
                                             stop_words='english')
        X_head = vectorizer_head.fit_transform(headlines)

        if term_freq == True:
            vectorizer_body = TfidfVectorizer(vocabulary=vocab, use_idf=use_idf,
                                             norm=norm, stop_words='english')
        else:
            vectorizer_body = CountVectorizer(vocabulary=vocab,
                                             stop_words='english')
        X_body = vectorizer_body.fit_transform(bodies)

        X = np.concatenate([X_head.toarray(), X_body.toarray()], axis=1)

        return X


    vocab = create_word_ngram_vocabulary(ngram_range=ngram_range, max_features=max_features,
                                         lemmatize=lemmatize, use_idf=use_idf, term_freq=term_freq, norm=norm,
                                         include_holdout=include_holdout)

    return get_features(vocab)

def get_head_body_tuples_test():
    data_path = "../../data"
    dataset = DataSet(data_path, "competition_test_bodies.csv", "competition_test_stances.csv", "test_set")

    h = []
    b = []
    for stance in dataset.stances:
        h.append(stance['Headline'])
        b.append(dataset.articles[int(stance['Body ID'])])

    return h, b


def get_head_body_tuples(include_holdout=False):
    # file paths
    '''
    data_path = "%s/data/fnc-1" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
    splits_dir = "%s/data/fnc-1/splits" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
    dataset = DataSet(data_path)
    '''
    data_path = "../../data"
    splits_dir = "../../splits"
    dataset = DataSet(data_path, "train_bodies.csv", "train_stances.csv", "train_set")

    def get_stances(dataset, folds, holdout):
        # Creates the list with a dict {'headline': ..., 'body': ..., 'stance': ...} for each
        # stance in the data set (except for holdout)
        stances = []
        for stance in dataset.stances:
            if stance['Body ID'] in holdout and include_holdout == True:
                stances.append(stance)
            for fold in folds:
                if stance['Body ID'] in fold:
                    stances.append(stance)

        return stances

    h = []
    b = []
    # create new vocabulary
    for stance in dataset.stances:
        h.append(stance['Headline'])
        b.append(dataset.articles[int(stance['Body ID'])])

    print("Stances length: " + str(len(dataset.stances)))


    # create the final lists with all the headlines and bodies of the set except for holdout
    # for stance in stances:
    #     h.append(stance['Headline'])
    #     b.append(dataset.articles[stance['Body ID']])

    return h, b

def create_word_ngram_vocabulary(ngram_range=(1,1), max_features=100, lemmatize=False, term_freq=False, norm='l1', use_idf=False, include_holdout=False):
    """
    Creates, returns and saves a vocabulary for (Count-)Vectorizer over all training and test data (holdout excluded) to create BoW
    methods. The method simplifies using the pipeline and later tests with feature creation for a single headline and body.
    This method will cause bleeding, since it also includes the test set.

    :param filename: a filename for the vocabulary
    :param ngram_range: the ngram range for the Vectorizer. Default is (1, 1) => unigrams
    :param max_features: the length of the vocabulary
    :return: the vocabulary
    """
    # file paths
    '''
    data_path = "%s/data/fnc-1" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
    splits_dir = "%s/data/fnc-1/splits" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
    features_dir = "%s/data/fnc-1/features" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

    dataset = DataSet(data_path)
    '''
    features_dir = "%s/features" % (path.dirname(path.dirname(path.dirname(__file__))))

    print("Calling create_word_ngram_vocabulary with ngram_range=("
          + str(ngram_range[0]) + ", " + str(ngram_range[1]) + "), max_features="
          + str(max_features) + ", lemmatize=" +  str(lemmatize) + ", term_freq=" + str(term_freq))
    def get_all_stopwords():
        stop_words_nltk = set(stopwords.words('english'))  # use set for faster "not in" check
        stop_words_sklearn = feature_extraction.text.ENGLISH_STOP_WORDS
        all_stop_words = stop_words_sklearn.union(stop_words_nltk)
        return all_stop_words

    def get_tokenized_lemmas_without_stopwords(s):
        all_stop_words = get_all_stopwords()
        return [normalize_word(t) for t in nltk.word_tokenize(s)
                if t not in string.punctuation and t.lower() not in all_stop_words]


    def train_vocabulary(head_and_body):
        # trains a CountVectorizer on all of the data except for holdout data
        if lemmatize == False:
            vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english', max_features=max_features)
            if term_freq == True:
                vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words='english', max_features=max_features, use_idf=use_idf, norm=norm)
        else:
            vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=max_features,
                                         tokenizer=get_tokenized_lemmas_without_stopwords)
            if term_freq == True:
                vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features,
                                             tokenizer=get_tokenized_lemmas_without_stopwords, use_idf=use_idf, norm=norm)
        vectorizer.fit_transform(head_and_body)
        vocab = vectorizer.vocabulary_
        return vocab

    def combine_head_and_body(headlines, bodies):
        head_and_body = [headline + " " + body for i, (headline, body) in
                         enumerate(zip(headlines, bodies))]
        return head_and_body


    # create filename for vocab
    vocab_file = "word_(" + str(ngram_range[0]) + "_" + str(ngram_range[1]) + ")-gram_" + str(max_features)
    if lemmatize == True:
        vocab_file += "_lemmatized"
    if term_freq == True:
        vocab_file += "_tf"
    if use_idf == True:
        vocab_file += "_idf"
    if include_holdout == True:
        vocab_file += "_holdout"
    vocab_file += "_" + norm + ".pickle"

    # if vocab already exists, just load and return it
    if (os.path.exists(features_dir + "/" + vocab_file)):
        with open(features_dir + "/" + vocab_file, 'rb') as handle:
            vocab = pickle.load(handle)
            print("Existing vocabulary found and load.")
            return vocab

    h, b = get_head_body_tuples(include_holdout=include_holdout)
    head_and_body = combine_head_and_body(h, b) # combine head and body
    vocab = train_vocabulary(head_and_body) # get vocabulary (features)

    # save the vocabulary as file
    with open(features_dir + "/" + vocab_file, 'wb') as handle:
        pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("vocab length: " + str(len(vocab)))
    return vocab


## Benjamins LSTM features:
def single_flat_LSTM_50d_100(headlines, bodies, name='train'):
    # Following the guide at https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    # see also documentation https://keras.io/layers/embeddings/

    """
    Improve on former LSTM features by dividing the tokens much better on the documents and evidences for a claim, in order to remove sparsitiy
    and add more useful information into the vectors.
    :param claims:
    :param evidences:
    :param orig_docs:
    :param fold:
    :return:
    """
    from keras_model.feature_helper.misc import create_embedding_lookup_pandas, \
        text_to_sequences_fixed_size, load_embedding_pandas

    #########################
    # PARAMETER DEFINITIONS #
    #########################
    method_name = "single_flat_LSTM_50d_100"
    # location path for features
    FEATURES_DIR = "%s/features/" % (path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
    PARAM_DICT_FILENAME = method_name+"_param_dict.pkl"

    param_dict = {
        "MAX_NB_WORDS": 50000,  # size of the vocabulary

        # sequence lengths
        "MAX_SEQ_LENGTH": 100, #1000

        # embedding specific values
        "EMBEDDING_DIM": 50,  # dimension of the GloVe embeddings
        "GLOVE_ZIP_FILE": 'glove.twitter.27B.zip',
        "GLOVE_FILE": 'glove.twitter.27B.50d.txt',

        # embedding file names
        "EMBEDDING_FILE": method_name+"_embedding.npy",

        # vocab file names
        "VOCAB_FILE": method_name+"_vocab.pkl",
    }


    ###############################################
    # GET VOCABULARY AND PREPARE EMBEDDING MATRIX #
    ###############################################

    # load GloVe embeddings
    GloVe_vectors = load_embedding_pandas(param_dict["GLOVE_ZIP_FILE"], param_dict["GLOVE_FILE"])

    # load all claims, orig_docs and evidences
    all_heads, all_bodies = word_ngrams.get_head_body_tuples(include_holdout=True)
    all = all_heads
    all.extend(all_bodies)


    # Comment out for clean ablation checks
    # add the unlabeled test data words to the BoW of test+train+holdout data
    h_unlbled_test, b_unlbled_test = word_ngrams.get_head_body_tuples_test()
    all.extend(h_unlbled_test)
    all.extend(b_unlbled_test)

    # create and save the embedding matrices for claims, orig_docs and evidences
    vocab = create_embedding_lookup_pandas(all, param_dict["MAX_NB_WORDS"], param_dict["EMBEDDING_DIM"],
                                           GloVe_vectors, param_dict["EMBEDDING_FILE"], param_dict["VOCAB_FILE"], init_zeros=False,
                                           add_unknown=True, rdm_emb_init=True, tokenizer=nltk.word_tokenize)

    # unload GloVe_vectors in order to make debugging possible
    del GloVe_vectors


    #################################################
    # Create sequences and embedding for the claims #
    #################################################
    print("Create sequences and embedding for the heads")

    # concatenated = []
    head_data = []
    body_data = []
    for i in range(len(headlines)):
        head_data.append(headlines[i])
        body_data.append(bodies[i])
        #원래는 concat(head, body)

    # replace tokens of claims by vocabulary ids - the ids refer to the index of the embedding matrix which holds the word embedding for this vocab word
    head_sequences = text_to_sequences_fixed_size(head_data, vocab, param_dict["MAX_SEQ_LENGTH"], save_full_text=False,
                                             take_full_claim=True)
    body_sequences = text_to_sequences_fixed_size(body_data, vocab, 300, save_full_text=False, take_full_claim=True)


    #################################################
    # SAVE PARAM_DICT AND CONCATENATE TRAINING DATA #
    #################################################

    # save param_dict
    if not os.path.exists(FEATURES_DIR+PARAM_DICT_FILENAME):
        with open(FEATURES_DIR+PARAM_DICT_FILENAME, 'wb') as f:
            pickle.dump(param_dict, f, pickle.HIGHEST_PROTOCOL)
        print("Save PARAM_DICT as " + FEATURES_DIR+PARAM_DICT_FILENAME)

    if not os.path.exists('./features/sequences_head_'+name+'.pkl'):
        with open('./features/sequences_head_'+name+'.pkl', 'wb') as f:
            pickle.dump(head_sequences, f, pickle.HIGHEST_PROTOCOL)
        print("Save sequences as ../features/sequences_head_"+name+".pkl")

    if not os.path.exists('./features/sequences_body_' + name + '.pkl'):
        with open('./features/sequences_body_' + name + '.pkl', 'wb') as f:
            pickle.dump(body_sequences, f, pickle.HIGHEST_PROTOCOL)
        print("Save sequences as ../features/sequences_body_" + name + ".pkl")

    return head_sequences, body_sequences

if __name__ == "__main__":
    ## make sequence train data

    h, b = get_head_body_tuples(include_holdout=False)
    sequences = single_flat_LSTM_50d_100(h, b, name='train')
    print(len(sequences))
    # create_word_ngram_vocabulary()

    ## test data
    h, b = get_head_body_tuples_test()
    # print(h[-1])

    sequences = single_flat_LSTM_50d_100(h, b, name='test')
    import pickle
    data = pickle.load(open('../../features/sequences_test.pkl', 'rb'))
    print(len(data[0]))
    print(len(data))
