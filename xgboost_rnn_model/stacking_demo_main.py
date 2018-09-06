import pickle as pkl

import numpy as np
import xgboost as xgb
from keras.models import load_model
from scipy.sparse import hstack
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import normalize

from news.ml_model.custom_vectorizer.gru_utils import nltk_corpus_download
from news.ml_model.custom_vectorizer.gru_vectorizer import GRUVectorizer
from news.ml_model.custom_vectorizer.xgb_vectorizer import XGBoostVectorizer
from news.ml_model.gru_models import GRULime
from news.ml_model.lime import LimeService
from news.ml_model.log_model_feature import LogModelFeature
from news.ml_model.utils import (
    get_head_body_tuples,
    get_head_body_tuples_test,
    get_y_labels
)
from news.ml_model.xgb_models import XGBoostLime


def read_dataset_test_head_body(datapath):
    head_test, body_test = get_head_body_tuples_test(data_path=datapath)
    dataset = {}
    idx = 0
    for head, body in zip(head_test, body_test):
        dataset[idx] = {'head' : head, 'body' : body }
        idx += 1
    return dataset


def read_dataset_train_head_body(datapath):
    head_train, body_train = get_head_body_tuples(data_path=datapath)
    dataset = {}
    idx = 0
    for head, body in zip(head_train, body_train):
        dataset[idx] = {'head' : head, 'body' : body }
        idx += 1
    return dataset


def load_GRU_model(model_path):
    return load_model(model_path+"/GRU_save.h5")


def get_GRU_predict(model_path, X_test, one_hot = True):
    print('load GRU model...')
    gru_model = load_GRU_model(model_path)
    print('Done.')
    if one_hot:
        return gru_model.predict(X_test)
    else:
        return np.argmax(gru_model.predict(X_test), axis=1)


def load_XGB_model(model_path, param):
    print('Load XGB model...')
    model = xgb.Booster()
    model.load_model(model_path + "/highest_model.model")
    print('Done.')
    return model


def load_XGB_params():
    param = {
        'max_depth': 6,  # (defalut 6)
        'colsample_bytree': 0.6,
        'subsample': 0.6,  # 트리마다의 관측 데이터 샘플링 비율, 값이 낮아질수록 under-fitting(default=1)
        'eta': 0.1,
        'objective': 'multi:softprob',
        'num_class': 4,
        'nthread': 16,
        'silent': 1,
        'eval_metric': 'mlogloss',
        'seed': 2017
    }
    return param


def load_pkl(file_path, filename):
    print('Loading {}/{}'.format(file_path, filename))
    pkl_file = pkl.load(open(file_path + "/" + filename, 'rb'))
    print('Load {}/{} finish!'.format(file_path, filename))
    return pkl_file


def get_train_X(data_path):
    train_tf_include_holdout_name = 'count_1st_train_include_test_combined.pkl'
    train_tfidf_include_holdout_name = 'tfidf_cos_train_include_holdout.pkl'
    train_nmf_name = 'nmf_200_cos_train_include_holdout.pkl'
    train_svd_name = 'svd_100_cos_train_include_holdout.pkl'
    train_doc2vec_name = 'glove200D_sum_head_body_train.pkl'

    train_tf_X = load_pkl(data_path, train_tf_include_holdout_name)
    train_tfidf_X = load_pkl(data_path, train_tfidf_include_holdout_name)
    train_svd_X = load_pkl(data_path, train_svd_name)
    train_nmf_X = load_pkl(data_path, train_nmf_name)
    # train_doc2vec_X = load_pkl(data_path, train_doc2vec_name)

    train_X = normalize(hstack((train_tf_X, train_tfidf_X,
                                train_svd_X, train_nmf_X), format='csr'))

    return train_X


def get_test_X(data_path):
    test_tf_include_holdout_name = 'count_1st_test_include_test_combined.pkl'
    test_tfidf_include_holdout_name = 'tfidf_cos_test_include_holdout.pkl'
    test_nmf_name = 'nmf_200_cos_test_include_holdout.pkl'
    test_svd_name = 'svd_100_cos_test_include_holdout.pkl'
    # test_doc2vec_name = 'glove200D_sum_head_body_test.pkl'

    test_tf_X = load_pkl(data_path, test_tf_include_holdout_name)
    test_tfidf_X = load_pkl(data_path, test_tfidf_include_holdout_name)
    test_svd_X = load_pkl(data_path, test_svd_name)
    test_nmf_X = load_pkl(data_path, test_nmf_name)
    # test_doc2vec_X = load_pkl(data_path, test_doc2vec_name)

    test_X = normalize(hstack((test_tf_X, test_tfidf_X,
                               test_svd_X, test_nmf_X), format='csr'))

    return test_X


def get_test_X_gru(data_path):
    X_test_GRU = pkl.load(open(data_path+'/sequences_test.pkl', 'rb'))
    return X_test_GRU

def load_stacking_logreg_model():
    return load_pkl('news/ml_model/pickled_data', 'stacking_logreg_model.pkl')


datapath = 'news/ml_model/data'
pickled_path = 'news/ml_model/pickled_data'
rnn_feat_path = 'news/ml_model/rnn_features'

gru_model_path = 'news/ml_model/saved_keras'
xgb_model_path = 'news/ml_model/xgboost_model_saved_file'

nltk_corpus_download()

xgb_params = load_XGB_params()
xgb_model = XGBoostLime(load_XGB_model(xgb_model_path, xgb_params))
gru_model = GRULime(load_GRU_model(gru_model_path))
stacking_log_model = load_stacking_logreg_model()

test_dataset = read_dataset_test_head_body(datapath)
test_features_xgb = get_test_X(pickled_path)
test_features_gru = get_test_X_gru(rnn_feat_path)
# print([test_features_gru[0]])
_, test_y = get_y_labels(pickled_path, one_hot=False)

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
xgb_feature_model = XGBoostVectorizer()
gru_feature_model = GRUVectorizer()
xgb_pipeline = Pipeline([
    ('xgb_feature', xgb_feature_model),
    ('xgb', xgb_model),
])
gru_pipeline = Pipeline([
    ('gru_feature', gru_feature_model),
    ('gru', gru_model),
])

log_model_feature = LogModelFeature(xgb_pipeline, gru_pipeline)

full_pipeline = Pipeline([
    ('log_feature', log_model_feature),
    ('log_reg', stacking_log_model)
])
