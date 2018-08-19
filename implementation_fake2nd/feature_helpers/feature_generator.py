from data_helper.Dataset_new import Dataset
from feature_helpers.feature_engineering import Feature_enginnering
from implementation.model_control import load_model, save_model
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import os
from time import time

base_data_path = os.path.dirname(os.path.dirname(__file__)) + "/data"
train_stance = base_data_path + "/train_stances.csv"
train_body = base_data_path + "/train_bodies.csv"
test_stance = base_data_path + "/competition_test_stances.csv"
test_body = base_data_path + "/competition_test_bodies.csv"


def make_tfidf_feature_5000(row_body_path, row_stance_path, head_save_path, body_save_path, stance_save_path,
                            model_save=True):
    if not os.path.exists(head_save_path) or not os.path.exists(body_save_path) \
            or not os.path.exists(stance_save_path):
        dataset = Dataset(row_body_path, row_stance_path)

        head, body, stance = dataset.read_combine()
        fe = Feature_enginnering(head, body, stance)
        # "tfidf_label_one_hot_train.pkl"
        # 'tfidf_body_feature_train.pkl'
        # 'tfidf_head_feature_train.pkl'
        fe.get_tfidf_vocab_5000()
        if not os.path.exists(head_save_path):
            fe.tfidf_train_head(head_save_path, model_save=model_save)
        if not os.path.exists(body_save_path):
            fe.tfidf_train_body(body_save_path, model_save=model_save)
        if not os.path.exists(stance_save_path):
            fe.tfidf_stance_save(stance_save_path, model_save=model_save)

    print('train_idf_5000 feature saved!')


def make_tfidf_feature_5000_holdout(row_body_path, row_stance_path, row_test_body_path, row_test_stance_path,
                                    head_save_path, body_save_path, stance_save_path, model_save=True):
    if not os.path.exists(head_save_path) or not os.path.exists(body_save_path) \
            or not os.path.exists(stance_save_path):
        dataset = Dataset(row_body_path, row_stance_path)

        head, body, stance = dataset.read_combine()
        fe = Feature_enginnering(head, body, stance)
        # "tfidf_label_one_hot_train.pkl"
        # 'tfidf_body_feature_train.pkl'
        # 'tfidf_head_feature_train.pkl'
        fe.get_tfidf_vocab_5000_holdout(row_test_body_path, row_test_stance_path)
        if not os.path.exists(head_save_path):
            fe.tfidf_train_head(head_save_path, model_save=model_save)
        if not os.path.exists(body_save_path):
            fe.tfidf_train_body(body_save_path, model_save=model_save)
        if not os.path.exists(stance_save_path):
            fe.tfidf_stance_save(stance_save_path, model_save=model_save)

    print('train_idf_5000 feature saved!')


def make_tfidf_feature_100(row_body_path, row_stance_path, head_save_path, body_save_path, stance_save_path,
                           model_save=True):
    if not os.path.exists(head_save_path) or not os.path.exists(body_save_path) \
            or not os.path.exists(stance_save_path):
        dataset = Dataset(row_body_path, row_stance_path)
        head, body, stance = dataset.read_combine()
        fe = Feature_enginnering(head, body, stance)
        # "tfidf_label_one_hot_train.pkl"
        # 'tfidf_body_feature_train.pkl'
        # 'tfidf_head_feature_train.pkl'
        fe.get_tfidf_vocab_100()
        if not os.path.exists(head_save_path):
            fe.tfidf_train_head(head_save_path, model_save=model_save)
        if not os.path.exists(body_save_path):
            fe.tfidf_train_body(body_save_path, model_save=model_save)
        if not os.path.exists(stance_save_path):
            fe.tfidf_stance_save(stance_save_path, model_save=model_save)

    print('train_idf_100 feature saved!')


def make_tfidf_feature_100_holdout(row_body_path, row_stance_path, row_test_body_path, row_test_stance_path,
                                   head_save_path, body_save_path, stance_save_path,
                                   model_save=True):
    if not os.path.exists(head_save_path) or not os.path.exists(body_save_path) \
            or not os.path.exists(stance_save_path):
        dataset = Dataset(row_body_path, row_stance_path)
        head, body, stance = dataset.read_combine()
        fe = Feature_enginnering(head, body, stance)
        # "tfidf_label_one_hot_train.pkl"
        # 'tfidf_body_feature_train.pkl'
        # 'tfidf_head_feature_train.pkl'
        fe.get_tfidf_vocab_100_holdout(row_test_body_path, row_test_stance_path)
        if not os.path.exists(head_save_path):
            fe.tfidf_train_head(head_save_path, model_save=model_save)
        if not os.path.exists(body_save_path):
            fe.tfidf_train_body(body_save_path, model_save=model_save)
        if not os.path.exists(stance_save_path):
            fe.tfidf_stance_save(stance_save_path, model_save=model_save)

    print('train_idf_100 feature saved!')


def make_tfidf_combined_feature_5000(row_body_path, row_stance_path, head_pkl, body_pkl, label_path):
    if not os.path.exists(body_pkl) or not os.path.exists(head_pkl):
        print('.pkl files not exist. We will make new TF-IDF .pkl vectors')
        make_tfidf_feature_5000(row_body_path, row_stance_path, head_pkl, body_pkl, label_path)
        print('.pkl make finish!')

    X_train_body = load_model(body_pkl)
    X_train_head = load_model(head_pkl)
    print('shape : ', X_train_body.shape, X_train_head.shape)

    X_train = np.concatenate((X_train_head.toarray(), X_train_body.toarray()), axis=1)
    return X_train


def make_tfidf_combined_feature_5000_holdout(row_body_path, row_stance_path, row_test_body_path, row_test_stance_path,
                                             head_pkl, body_pkl, label_path):
    if not os.path.exists(body_pkl) or not os.path.exists(head_pkl):
        print('.pkl files not exist. We will make new TF-IDF .pkl vectors')
        make_tfidf_feature_5000_holdout(row_body_path, row_stance_path, row_test_body_path, row_test_stance_path,
                                head_pkl, body_pkl, label_path)
        print('.pkl make finish!')

    X_train_body = load_model(body_pkl)
    X_train_head = load_model(head_pkl)
    print('shape : ', X_train_body.shape, X_train_head.shape)

    X_train = np.concatenate((X_train_head.toarray(), X_train_body.toarray()), axis=1)
    return X_train


def make_tfidf_combined_feature_100(row_body_path, row_stance_path, head_pkl, body_pkl, label_path):
    if not os.path.exists(body_pkl) or not os.path.exists(head_pkl):
        print('.pkl files not exist. We will make new TF-IDF .pkl vectors')
        make_tfidf_feature_100(row_body_path, row_stance_path, head_pkl, body_pkl, label_path)
        print('.pkl make finish!')

    X_train_body = load_model(body_pkl)
    X_train_head = load_model(head_pkl)
    print('shape : ', X_train_body.shape, X_train_head.shape)

    X_train = np.concatenate((X_train_head.toarray(), X_train_body.toarray()), axis=1)
    return X_train


def make_tfidf_combined_feature_100_holdout(row_body_path, row_stance_path, row_test_body_path, row_test_stance_path,
                                            head_pkl, body_pkl, label_path):
    if not os.path.exists(body_pkl) or not os.path.exists(head_pkl):
        print('.pkl files not exist. We will make new TF-IDF .pkl vectors')
        make_tfidf_feature_100_holdout(row_body_path, row_stance_path, row_test_body_path, row_test_stance_path,
                               head_pkl, body_pkl, label_path)
        print('.pkl make finish!')

    X_train_body = load_model(body_pkl)
    X_train_head = load_model(head_pkl)
    print('shape : ', X_train_body.shape, X_train_head.shape)

    X_train = np.concatenate((X_train_head.toarray(), X_train_body.toarray()), axis=1)
    return X_train


def load_tfidf_y(pkl_path):
    return load_model(pkl_path)


def make_tfidf_cos_feature_5000(row_body_path, row_stance_path, head_save_path, body_save_path, stance_save_path,
                                tfidf_cos_path,
                                model_save=True):
    if not os.path.exists(head_save_path) or not os.path.exists(body_save_path) \
            or not os.path.exists(stance_save_path) or not os.path.exists(tfidf_cos_path):
        dataset = Dataset(row_body_path, row_stance_path)
        head, body, stance = dataset.read_combine()
        fe = Feature_enginnering(head, body, stance)
        # "tfidf_label_one_hot_train.pkl"
        # 'tfidf_body_feature_train.pkl'
        # 'tfidf_head_feature_train.pkl'
        fe.get_tfidf_vocab_5000()
        if not os.path.exists(head_save_path):
            fe.tfidf_train_head(head_save_path, model_save=model_save)
        if not os.path.exists(body_save_path):
            fe.tfidf_train_body(body_save_path, model_save=model_save)
        if not os.path.exists(stance_save_path):
            fe.tfidf_stance_save(stance_save_path, model_save=model_save)
        if not os.path.exists(tfidf_cos_path):
            fe.tfidf_cos_save(head_save_path, body_save_path, tfidf_cos_path, model_save=model_save)

    print('train_idf_5000 features saved!')


def make_tfidf_cos_feature_5000_holdout(row_body_path, row_stance_path, row_test_body_path, row_test_stance_path,
                                        head_save_path, body_save_path, stance_save_path, tfidf_cos_path,
                                        model_save=True):
    if not os.path.exists(head_save_path) or not os.path.exists(body_save_path) \
            or not os.path.exists(stance_save_path) or not os.path.exists(tfidf_cos_path):
        dataset = Dataset(row_body_path, row_stance_path)
        head, body, stance = dataset.read_combine()
        fe = Feature_enginnering(head, body, stance)
        # "tfidf_label_one_hot_train.pkl"
        # 'tfidf_body_feature_train.pkl'
        # 'tfidf_head_feature_train.pkl'
        fe.get_tfidf_vocab_5000_holdout(row_test_body_path, row_test_stance_path)
        if not os.path.exists(head_save_path):
            fe.tfidf_train_head(head_save_path, model_save=model_save)
        if not os.path.exists(body_save_path):
            fe.tfidf_train_body(body_save_path, model_save=model_save)
        if not os.path.exists(stance_save_path):
            fe.tfidf_stance_save(stance_save_path, model_save=model_save)
        if not os.path.exists(tfidf_cos_path):
            fe.tfidf_cos_save(head_save_path, body_save_path, tfidf_cos_path, model_save=model_save)

    print('train_idf_5000 features saved!')

def make_tfidf_cos_feature_5000_holdout_test(row_body_path, row_stance_path, row_test_body_path, row_test_stance_path,
                                        head_save_path, body_save_path, stance_save_path, tfidf_cos_path,
                                        model_save=True):
    if not os.path.exists(head_save_path) or not os.path.exists(body_save_path) \
            or not os.path.exists(stance_save_path) or not os.path.exists(tfidf_cos_path):
        dataset = Dataset(row_body_path, row_stance_path)
        head, body, stance = dataset.read_combine()
        fe = Feature_enginnering(head, body, stance)
        # "tfidf_label_one_hot_train.pkl"
        # 'tfidf_body_feature_train.pkl'
        # 'tfidf_head_feature_train.pkl'
        fe.get_tfidf_vocab_5000_holdout(row_test_body_path, row_test_stance_path)
        if not os.path.exists(head_save_path):
            fe.tfidf_train_head(head_save_path, model_save=model_save)
        if not os.path.exists(body_save_path):
            fe.tfidf_train_body(body_save_path, model_save=model_save)
        if not os.path.exists(stance_save_path):
            fe.tfidf_stance_save(stance_save_path, model_save=model_save)
        if not os.path.exists(tfidf_cos_path):
            fe.tfidf_cos_save(head_save_path, body_save_path, tfidf_cos_path, model_save=model_save)

    print('train_idf_5000 features saved!')

def make_tfidf_cos_feature_100(row_body_path, row_stance_path, head_save_path, body_save_path, stance_save_path,
                               tfidf_cos_path,
                               model_save=True):
    if not os.path.exists(head_save_path) or not os.path.exists(body_save_path) \
            or not os.path.exists(stance_save_path) or not os.path.exists(tfidf_cos_path):
        dataset = Dataset(row_body_path, row_stance_path)
        head, body, stance = dataset.read_combine()
        fe = Feature_enginnering(head, body, stance)
        # "tfidf_label_one_hot_train.pkl"
        # 'tfidf_body_feature_train.pkl'
        # 'tfidf_head_feature_train.pkl'
        fe.get_tfidf_vocab_100()
        print('cos_path : ', tfidf_cos_path)
        if not os.path.exists(head_save_path):
            fe.tfidf_train_head(head_save_path, model_save=model_save)
        if not os.path.exists(body_save_path):
            fe.tfidf_train_body(body_save_path, model_save=model_save)
        if not os.path.exists(stance_save_path):
            fe.tfidf_stance_save(stance_save_path, model_save=model_save)
        if not os.path.exists(tfidf_cos_path):
            fe.tfidf_cos_save(head_save_path, body_save_path, tfidf_cos_path, model_save=model_save)

    print('train_idf_cos_100 features saved!')


def make_tfidf_cos_feature_100_holdout(row_body_path, row_stance_path, row_test_body_path, row_test_stance_path,
                                       head_save_path, body_save_path, stance_save_path, tfidf_cos_path,
                                       model_save=True):
    if not os.path.exists(head_save_path) or not os.path.exists(body_save_path) \
            or not os.path.exists(stance_save_path) or not os.path.exists(tfidf_cos_path):
        dataset = Dataset(row_body_path, row_stance_path)
        head, body, stance = dataset.read_combine()
        fe = Feature_enginnering(head, body, stance)
        # "tfidf_label_one_hot_train.pkl"
        # 'tfidf_body_feature_train.pkl'
        # 'tfidf_head_feature_train.pkl'
        fe.get_tfidf_vocab_100_holdout(row_test_body_path, row_test_stance_path)
        print('cos_path : ', tfidf_cos_path)
        if not os.path.exists(head_save_path):
            fe.tfidf_train_head(head_save_path, model_save=model_save)
        if not os.path.exists(body_save_path):
            fe.tfidf_train_body(body_save_path, model_save=model_save)
        if not os.path.exists(stance_save_path):
            fe.tfidf_stance_save(stance_save_path, model_save=model_save)
        if not os.path.exists(tfidf_cos_path):
            fe.tfidf_cos_save(head_save_path, body_save_path, tfidf_cos_path, model_save=model_save)

    print('train_idf_cos_100 features saved!')


def make_tfidf_combined_feature_cos_5000(row_body_path, row_stance_path,
                                                 head_pkl, body_pkl, label_path, tfidf_cos_path):
    if not os.path.exists(body_pkl) or not os.path.exists(head_pkl) \
            or not os.path.exists(label_path) or not os.path.exists(tfidf_cos_path):
        print('.pkl files not exist. We will make new TF-IDF .pkl vectors')
        make_tfidf_cos_feature_5000(row_body_path, row_stance_path, head_pkl, body_pkl, label_path, tfidf_cos_path)
        print('.pkl make finish!')

    X_train_body = load_model(body_pkl)
    X_train_head = load_model(head_pkl)
    X_train_cos = load_model(tfidf_cos_path)

    print('shape : ', X_train_body.shape, X_train_head.shape)

    print(X_train_cos.shape)
    X_train = np.concatenate((X_train_head.toarray(), X_train_body.toarray()), axis=1)
    X_train = np.concatenate((X_train, X_train_cos), axis=1)

    return X_train


def make_tfidf_combined_feature_cos_5000_holdout(row_body_path, row_stance_path, row_test_body_path, row_test_stance_path,
                                                 head_pkl, body_pkl, label_path, tfidf_cos_path):
    if not os.path.exists(body_pkl) or not os.path.exists(head_pkl) \
            or not os.path.exists(label_path) or not os.path.exists(tfidf_cos_path):
        print('.pkl files not exist. We will make new TF-IDF .pkl vectors')
        make_tfidf_cos_feature_5000_holdout(row_body_path, row_stance_path, row_test_body_path, row_test_stance_path,
                                            head_pkl, body_pkl, label_path, tfidf_cos_path)
        print('.pkl make finish!')

    X_train_body = load_model(body_pkl)
    X_train_head = load_model(head_pkl)
    X_train_cos = load_model(tfidf_cos_path)

    print('shape : ', X_train_body.shape, X_train_head.shape)

    print(X_train_cos.shape)
    X_train = np.concatenate((X_train_head.toarray(), X_train_body.toarray()), axis=1)
    X_train = np.concatenate((X_train, X_train_cos), axis=1)

    return X_train

def make_tfidf_combined_feature_cos_5000_holdout_test(row_body_path, row_stance_path,
                                                 head_pkl, body_pkl, label_path, tfidf_cos_path):
    if not os.path.exists(body_pkl) or not os.path.exists(head_pkl) \
            or not os.path.exists(label_path) or not os.path.exists(tfidf_cos_path):
        print('.pkl files not exist. We will make new TF-IDF .pkl vectors')
        make_tfidf_cos_feature_5000(row_body_path, row_stance_path,
                                            head_pkl, body_pkl, label_path, tfidf_cos_path)
        print('.pkl make finish!')

    X_train_body = load_model(body_pkl)
    X_train_head = load_model(head_pkl)
    X_train_cos = load_model(tfidf_cos_path)

    print('shape : ', X_train_body.shape, X_train_head.shape)

    print(X_train_cos.shape)
    X_train = np.concatenate((X_train_head.toarray(), X_train_body.toarray()), axis=1)
    X_train = np.concatenate((X_train, X_train_cos), axis=1)

    return X_train

def make_tfidf_combined_feature_cos_100(row_body_path, row_stance_path, head_pkl, body_pkl, label_path,
                                        tfidf_cos_path):
    print(tfidf_cos_path)
    if not os.path.exists(body_pkl) or not os.path.exists(head_pkl) or \
            not os.path.exists(label_path) or not os.path.exists(tfidf_cos_path):
        print('.pkl files not exist. We will make new TF-IDF .pkl vectors')
        make_tfidf_cos_feature_100(row_body_path, row_stance_path, head_pkl, body_pkl, label_path, tfidf_cos_path)
        print('.pkl make finish!')

    X_train_body = load_model(body_pkl)
    X_train_head = load_model(head_pkl)
    X_train_cos = load_model(tfidf_cos_path)

    print('shape : ', X_train_body.shape, X_train_head.shape)

    print(X_train_cos.shape)
    X_train = np.concatenate((X_train_head.toarray(), X_train_body.toarray()), axis=1)
    X_train = np.concatenate((X_train, X_train_cos), axis=1)

    return X_train


def make_tfidf_combined_feature_cos_100_holdout(row_body_path, row_stance_path, row_test_body_path, row_test_stance_path,
                                                head_pkl, body_pkl, label_path, tfidf_cos_path):
    print(tfidf_cos_path)
    if not os.path.exists(body_pkl) or not os.path.exists(head_pkl) or \
            not os.path.exists(label_path) or not os.path.exists(tfidf_cos_path):
        print('.pkl files not exist. We will make new TF-IDF .pkl vectors')
        make_tfidf_cos_feature_100_holdout(row_body_path, row_stance_path, row_test_body_path, row_test_stance_path,
                                           head_pkl, body_pkl, label_path, tfidf_cos_path)
        print('.pkl make finish!')

    X_train_body = load_model(body_pkl)
    X_train_head = load_model(head_pkl)
    X_train_cos = load_model(tfidf_cos_path)

    print('shape : ', X_train_body.shape, X_train_head.shape)

    print(X_train_cos.shape)
    X_train = np.concatenate((X_train_head.toarray(), X_train_body.toarray()), axis=1)
    X_train = np.concatenate((X_train, X_train_cos), axis=1)

    return X_train


def make_NMF_300_feature(row_body_path, row_stance_path, head_tfidf_pkl, body_tfidf_pkl, label_path,
                         save_nmf_model_path, save_head_path, save_body_path, cos_dist=False):
    if not os.path.exists(head_tfidf_pkl) or not os.path.exists(body_tfidf_pkl) \
            or not os.path.exists(label_path):
        make_tfidf_feature_5000(row_body_path, row_stance_path, head_tfidf_pkl, body_tfidf_pkl, label_path,
                                model_save=True)

    X_tfidf_body = load_model(body_tfidf_pkl)
    X_tfidf_head = load_model(head_tfidf_pkl)

    if not os.path.exists(save_nmf_model_path):
        X_all = np.concatenate((X_tfidf_head.toarray(), X_tfidf_body.toarray()), axis=0)
        print('fit NMF topic model')
        t0 = time()
        nmf = NMF(n_components=300, random_state=1, alpha=.1)
        nmf.fit(X_all)
        print('done in {}'.format(time() - t0))
        save_model(save_nmf_model_path, nmf)

    nmf = load_model(save_nmf_model_path)

    if not os.path.exists(save_head_path) or not os.path.exists(save_body_path):
        nmf_head_matrix = nmf.transform(X_tfidf_head)
        nmf_body_matrix = nmf.transform(X_tfidf_body)
        save_model(save_head_path, nmf_head_matrix)
        print('saved model {}'.format(save_head_path))
        save_model(save_body_path, nmf_body_matrix)
        print('saved model {}'.format(save_body_path))

    nmf_head_matrix = load_model(save_head_path)
    nmf_body_matrix = load_model(save_body_path)
    if not cos_dist:
        return np.concatenate((nmf_head_matrix, nmf_body_matrix), axis=1)
    else:
        X = []
        for i in range(len(nmf_head_matrix)):
            X_head = np.array(nmf_head_matrix[i]).reshape((1, -1))
            X_body = np.array(nmf_body_matrix[i]).reshape((1, -1))
            cos = cosine_distances(X_head, X_body).flatten()
            X.append(cos.tolist())
        X = np.array(X)
        X_train = np.concatenate((nmf_head_matrix, nmf_body_matrix), axis=1)
        X = np.concatenate((X_train, X), axis=1)
        return X


if __name__ == "__main__":
    body_path = "../data/train_bodies.csv"
    stance_path = "../data/train_stances.csv"
    head_pkl = "../pickled_model/tfidf_head_feature.pkl"
    body_pkl = "../pickled_model/tfidf_body_feature.pkl"

    # make_tfidf_feature_add_holdout_data(body_path, stance_path, True)
