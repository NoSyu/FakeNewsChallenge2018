from data_helper.dataset_new import Dataset
from features.feature_engineering import Feature_enginnering
from implementation.model_control import load_model
import numpy as np
import os
base_data_path = os.path.dirname(os.path.dirname(__file__)) + "/data"
train_stance = base_data_path + "/train_stances.csv"
train_body = base_data_path + "/train_bodies.csv"
test_stance = base_data_path + "/competition_test_stances.csv"
test_body = base_data_path + "/competition_test_bodies.csv"


def make_tfidf_feature(row_body_path, row_stance_path, head_save_path, body_save_path, stance_save_path, model_save=True):

    dataset = Dataset(row_body_path, row_stance_path)
    head, body, stance = dataset.read_combine()
    fe = Feature_enginnering(head, body, stance)
    # "tfidf_label_one_hot_train.pkl"
    # 'tfidf_body_feature_train.pkl'
    # 'tfidf_head_feature_train.pkl'
    fe.get_tfidf_vocab()
    fe.tfidf_train_head(head_save_path, model_save=model_save)
    fe.tfidf_train_body(body_save_path, model_save=model_save)
    fe.tfidf_stance_save(stance_save_path, model_save=model_save)

    print('train_idf feature saved!')


# def make_tfidf_feature_add_holdout_data(body_path, stance_path, model_save=True):
#     dataset = Dataset(body_path, stance_path)
#     head, body, stance = dataset.read_combine()
#     fe = Feature_enginnering(head, body, stance)
#     fe.read_more_data(test_body, test_stance)
#
#     fe.tfidf_stance_save("tfidf_label_one_hot_train_holdout.pkl", model_save=model_save)
#     fe.get_tfidf_vocab()
#     fe.tfidf_train_body('tfidf_body_feature_train_holdout.pkl', model_save=model_save)
#     fe.tfidf_train_head('tfidf_head_feature_train_holdout.pkl', model_save=model_save)
#     print('train_holdout_idf feature saved!')

def make_tfidf_combined_feature(row_body_path, row_stance_path, head_pkl, body_pkl, label_path):


    if not os.path.exists(body_pkl) or not os.path.exists(head_pkl):
        print('.pkl files not exist. We will make new TF-IDF .pkl vectors')
        make_tfidf_feature(row_body_path, row_stance_path, head_pkl, body_pkl, label_path)
        print('.pkl make finish!')

    X_train_body = load_model(body_pkl)
    X_train_head = load_model(head_pkl)
    print('shape : ', X_train_body.shape, X_train_head.shape)

    X_train = np.concatenate((X_train_head.toarray(), X_train_body.toarray()), axis=1)
    return X_train

def load_tfidf_y(pkl_path):
    return load_model(pkl_path)

if __name__ == "__main__":
    body_path = "../data/train_bodies.csv"
    stance_path = "../data/train_stances.csv"
    head_pkl = "../pickled_model/tfidf_head_feature.pkl"
    body_pkl = "../pickled_model/tfidf_body_feature.pkl"

    # make_tfidf_feature_add_holdout_data(body_path, stance_path, True)


