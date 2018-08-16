from data_helper.Dataset_new import Dataset
import zipfile
import numpy as np
import nltk
import pickle
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import os.path as path
from time import time
import os

method_name = "single_flat_LSTM_50d_100"
PARAM_DICT_FILENAME = method_name + "_param_dict.pkl"
EMBEDDINGS_DIR = "{}/glove_data/".format(path.dirname(path.dirname(path.abspath(__file__))))
FEATURES_DIR = "{}/features/".format(path.dirname(path.dirname(path.abspath(__file__))))

param_dict={
    "MAX_NB_WORDS" : 50000,
    "MAX_SEQ_LENGTH" : 100,

    "EMBEDDING_DIM" : 50,
    "GLOVE_ZIP_FILE": 'glove.twitter.27B.zip',
    "GLOVE_FILE" : 'glove.twitter.27B.50d.txt',

    "EMBEDDING_FILE" : method_name + "_embedding.npy",

    "VOCAB_FILE" : method_name + "_vocab.pkl"
}

def load_embedding_pandas(zip_file, file, type="w2v"):
    import pandas as pd
    import csv
    print('read embedding...')
    t0 = time()

    with zipfile.ZipFile(EMBEDDINGS_DIR+zip_file) as z:
        if type == "w2v":
            embedding = pd.read_table(z.open(file), sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE, skiprows=1)
        else:
            embedding = pd.read_table(z.open(file), sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
        print('Found {} word vectors in GloVe embeddings.'.format(len(embedding.index)))
    print('read finish, loading time : {}'.format(time()-t0))
    return embedding

def create_embedding_lookup_pandas(text_list, max_nb_words, embedding_dim, embedding,
                                   embedding_lookup_name, embedding_vocab_name, rdm_emb_init=False,
                                   add_unknown=False, tokenizer=None, init_zeros = False):
    if not path.exists(FEATURES_DIR + embedding_lookup_name) or not path.exists(FEATURES_DIR + embedding_vocab_name):
        vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words=None, tokenizer=tokenizer, max_features=max_nb_words,
                                     use_idf=True)
        vectorizer.fit_transform(text_list)
        vocab = vectorizer.vocabulary_

        for word in vocab.keys():
            vocab[word] += 1

        if add_unknown == True:
            max_index = max(vocab.values())
            vocab["UNKNOWN"] = max_index+1
        if rdm_emb_init == True:
            embedding_lookup = np.random.random((len(vocab) + 1, embedding_dim))
            zero_vec = np.zeros((embedding_dim))
            embedding_lookup[0] = zero_vec
        else:
            embedding_lookup = np.zeros((len(vocab) + 1, embedding_dim))

        if init_zeros == False:
            for word, i in vocab.items():
                if word == "UNKNOWN":
                    embedding_vector = np.random.uniform(low=-0.05, high=0.05, size=embedding_dim)
                else:
                    try:
                        embedding_vector = embedding.loc[word].values()
                    except KeyError:
                        continue
                if embedding_vector is not None:
                    embedding_lookup[i] = embedding_vector

        np.save(FEATURES_DIR+embedding_lookup_name, embedding_lookup)

        with open(FEATURES_DIR + embedding_vocab_name, 'wb') as f:
            pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

        print("Embedding lookup table shape for " + embedding_lookup_name + " is: " + str(embedding_lookup.shape))
    else:
        with open(FEATURES_DIR + embedding_vocab_name, 'rb') as f:
            vocab = pickle.load(f)
    print("Vocab size for " + embedding_vocab_name + " is: " + str(len(vocab)))

    return vocab

def text_to_sequences_fixed_size(texts, vocab, MAX_SENT_LENGTH, save_full_text=False, take_full_claim=False):
    data = np.zeros((len(texts), MAX_SENT_LENGTH), dtype='int32')

    claims = []
    if take_full_claim == False:
        for claim in texts:
            claim_sents = nltk.sent_tokenize(claim)
            word_count_fct = lambda sentence : len(nltk.word_tokenize(sentence))
            # 가장 단어가 많은 문장을 claims에 넣음
            claims.append(max(claim_sents, key=word_count_fct))
    else:
        claims = texts

    data_string_dict = {}
    for j, claim in tqdm(enumerate(claims)):
        claim_tokens = nltk.word_tokenize(claim.lower())

        data_string = ""
        if save_full_text == True:
            for token in claim_tokens:
                data_string += token + " "
            data_string = data_string[:-1]
            data_string_dict[j] = data_string

        for i, token in enumerate(claim_tokens):
            if i < MAX_SENT_LENGTH:
                # vocab에 token이 없으면 "UNKNOWN"을 가져옴("UNKNOWN"이 디폴트 값)
                index = vocab.get(token, "UNKNOWN")
                if index == "UNKNOWN":
                    index = vocab.get(index, None)
                if index != None:
                    data[j, i] = index

    if save_full_text == True:
        return data, data_string_dict
    else:
        return data

def single_flat_LSTM_50d_100(body_path, stance_path, mode):

    GloVe_vectors = load_embedding_pandas(param_dict['GLOVE_ZIP_FILE'], param_dict['GLOVE_FILE'], type="w2v")
    print(GloVe_vectors[:5])
    d_set = Dataset(body_path, stance_path)
    head, body, one_hot_label = d_set.read_combine()
    all = head.tolist()
    all.extend(body.tolist())

    vocab = create_embedding_lookup_pandas(all, param_dict["MAX_NB_WORDS"], param_dict["EMBEDDING_DIM"], GloVe_vectors,
                                           param_dict["EMBEDDING_FILE"], param_dict["VOCAB_FILE"], init_zeros=False,
                                           add_unknown=True, rdm_emb_init=True, tokenizer=nltk.word_tokenize)

    del GloVe_vectors

    concatenated = []
    for i in range(len(head)):
        concatenated.append(head[i] + ". " + body[i])
    sequences = text_to_sequences_fixed_size(concatenated, vocab, param_dict["MAX_SEQ_LENGTH"],
                                             save_full_text=False, take_full_claim=True)

    if mode == 'train':
        with open(FEATURES_DIR + PARAM_DICT_FILENAME, 'wb') as f:
            pickle.dump(param_dict, f, pickle.HIGHEST_PROTOCOL)
        print("Save PARAM_DICT as " + FEATURES_DIR + PARAM_DICT_FILENAME)

    return sequences

def get_sequence_data(body_path, stance_path, mode):
    # base_data_path = os.path.dirname(os.path.dirname(__file__)) + "/data"
    base_feat_path = os.path.dirname(os.path.dirname(__file__)) + "/features"
    if mode == 'train':
        if not path.exists(base_feat_path+"/single_flat_LSTM_50d_100_embedding.npy") or \
            not path.exists(base_feat_path+"/single_flat_LSTM_50d_100_train.npy"):
            sequences = single_flat_LSTM_50d_100(body_path, stance_path, mode)
            np.save(base_feat_path+"/single_flat_LSTM_50d_100_train.npy", sequences)
    elif mode == 'test':
        if not path.exists(base_feat_path + "/single_flat_LSTM_50d_100_embedding.npy") or \
                not path.exists(base_feat_path + "/single_flat_LSTM_50d_100_test.npy"):
            sequences = single_flat_LSTM_50d_100(body_path, stance_path, mode)
            np.save(base_feat_path + "/single_flat_LSTM_50d_100_"+mode+".npy", sequences)

    sequences = np.load(base_feat_path+"/single_flat_LSTM_50d_100_"+mode+".npy")

    return sequences


if __name__ == "__main__":
    import tensorflow as tf
    base_data_path = os.path.dirname(os.path.dirname(__file__)) + "/data"
    base_feat_path = os.path.dirname(os.path.dirname(__file__)) + "/features"
    train_stance = base_data_path + "/train_stances.csv"
    train_body = base_data_path + "/train_bodies.csv"
    test_stance = base_data_path + "/competition_test_stances.csv"
    test_body = base_data_path + "/competition_test_bodies.csv"


    # sequences, vocab = single_flat_LSTM_50d_100(train_body, train_stance)
    # if not path.exists(base_data_path+"/single_flat_LSTM_50d_100_train.npy"):
    #     np.save(base_feat_path+"/single_flat_LSTM_50d_100_train.npy", sequences)
    # print(sequences[:10])


    embedding = np.load(base_feat_path+"/single_flat_LSTM_50d_100_embedding.npy")
    sequences = np.load(base_feat_path + "/single_flat_LSTM_50d_100_train.npy")
    print(sequences[:10])


    sess = tf.InteractiveSession()
    params = tf.constant([[1, 2, 3], [10, 20, 30], [10, 20, 10],
                          [4, 5, 6], [-4, -5, -6], [-10, -20, -30], [-10, -10, 30]])
    ids = tf.constant([[5, 1, 0, 3], [6, 2, 0, 4]])

    print(tf.nn.embedding_lookup(params, ids).eval())



    print(tf.nn.embedding_lookup(params, ids).eval().shape)
    # print(embedding)
    # print(embedding.index_col)

    # GloVe_vectors = load_embedding_pandas(param_dict['GLOVE_ZIP_FILE'], param_dict['GLOVE_FILE'], type="w2v")
    # print(GloVe_vectors)
    # # vocab = pickle.load(open(base_feat_path+"/"+param_dict["VOCAB_FILE"], 'rb'))
    # embedded_seq = []


    # for s in tqdm(sequences):
    #     seq_tmp = [embedding[word_idx] for word_idx in s]
    #     embedded_seq.append(np.array(seq_tmp))
    #
    # print(len(embedded_seq))
