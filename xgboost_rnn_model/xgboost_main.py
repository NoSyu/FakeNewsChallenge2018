from Tree_models.model.XGBoost import XGBoost
from Tree_models.utils.get_input_datas import get_y_labels
# from Tree_models.utils.score import report_score
import pickle as pkl
from scipy.sparse import hstack
import xgboost as xgb
from sklearn.preprocessing import normalize
import numpy as np
# from Tree_models.utils.score import report_score

def load_pkl(file_path, filename):
    print('Loading {}/{}'.format(file_path, filename))
    pkl_file = pkl.load(open(file_path + "/" + filename, 'rb'))
    print('Load {}/{} finish!'.format(file_path, filename))
    return pkl_file

param = {
        'max_depth' : 6,#(defalut 6)
        'colsample_bytree' : 0.6,
        'subsample' : 1.0, # 트리마다의 관측 데이터 샘플링 비율, 값이 낮아질수록 under-fitting(default=1)
        'eta' : 0.1,
        'objective' : 'multi:softprob',
        'num_class' : 4,
        'nthread' : 16,
        'silent' : 1,
        'eval_metric' : 'mlogloss',
        'seed' : 2017
    }
num_round = 3000
verbose_eval = 5
early_stopping_rounds = 100

features_path = '../saved_model'
pickled_path = '../pickled_data'
save_model_path = '../xgboost_model_saved_file'

# train_tf_name = 'count_1st_train_combined.pkl'
train_tf_include_holdout_name = 'count_1st_train_include_test_combined.pkl'
# train_tfidf_name = 'tfidf_feat_1st_cos_train.pkl'
train_tfidf_include_holdout_name = 'tfidf_cos_train_include_holdout.pkl'
train_nmf_name = 'nmf_200_cos_train_include_holdout.pkl'
train_svd_name = 'svd_100_cos_train_include_holdout.pkl'
train_doc2vec_name = 'glove200D_sum_head_body_train.pkl'
# train_sent_name = 'senti_train_cos.pkl'

# test_tf_name = 'count_1st_test_combined.pkl'
test_tf_include_holdout_name = 'count_1st_test_include_test_combined.pkl'
# test_tfidf_name = 'tfidf_feat_1st_cos_test.pkl'
test_tfidf_include_holdout_name = 'tfidf_cos_test_include_holdout.pkl'
test_nmf_name = 'nmf_200_cos_test_include_holdout.pkl'
test_svd_name = 'svd_100_cos_test_include_holdout.pkl'
test_doc2vec_name = 'glove200D_sum_head_body_test.pkl'
# test_sent_name = 'senti_test_cos.pkl'

train_tf_X = load_pkl(pickled_path, train_tf_include_holdout_name)
train_tfidf_X = load_pkl(features_path, train_tfidf_include_holdout_name)
train_svd_X = load_pkl(features_path, train_svd_name)
train_nmf_X = load_pkl(features_path, train_nmf_name)
train_doc2vec_X = load_pkl(pickled_path, train_doc2vec_name)
# train_sentiment_X = load_pkl(pickled_path, train_sent_name)

train_X = normalize(hstack((train_tf_X, train_tfidf_X, train_svd_X, train_nmf_X), format='csr'))
# train_X = normalize(hstack((train_tf_X, train_tfidf_X, train_svd_X, train_nmf_X), format='csr'))

test_tf_X = load_pkl(pickled_path, test_tf_include_holdout_name)
test_tfidf_X = load_pkl(features_path, test_tfidf_include_holdout_name)
test_svd_X = load_pkl(features_path, test_svd_name)
test_nmf_X = load_pkl(features_path, test_nmf_name)
test_doc2vec_X = load_pkl(pickled_path, test_doc2vec_name)
# test_sentiment_X = load_pkl(pickled_path, test_sent_name)

test_X = normalize(hstack((test_tf_X, test_tfidf_X, test_svd_X, test_nmf_X), format='csr'))
# test_X = normalize(hstack((test_tf_X, test_tfidf_X, test_svd_X, test_nmf_X), format='csr'))

train_y, test_y = get_y_labels(pickled_path, one_hot=False)

dtrain = xgb.DMatrix(train_X, label=train_y)
dtest = xgb.DMatrix(test_X, label=test_y)

model = XGBoost(max_depth=param['max_depth'], colsample_bytree=param['colsample_bytree'],
              subsample=param['subsample'], eta=param['eta'], objective=param['objective'],
              num_class=param['num_class'], nthread=param['nthread'], silent=param['silent'],
              eval_metric=param['eval_metric'], seed=param['seed'])

# model.train(train_data=dtrain, test_data=dtest, num_round=num_round,
#           verbose_eval=verbose_eval, early_stopping_rounds=early_stopping_rounds)
# model.save_model(model_path=save_model_path, model_name='use_tf_tfidfcos_svdcos100_nmfcos200_doc2vec200_sentiment_include_holdout_'
#                                                         +str(num_round)+'_eta'+str(param['eta'])+'.model')
# model.load_model(model_path=save_model_path, model_name='use_tf_tfidfcos_svdcos100_nmfcos200_include_holdout_3000_eta0.1.model')

model.load_model(model_path=save_model_path, model_name='highest_model.model')

predicted = model.predict(actual=test_y, test_data=dtest, feature_size = test_X.shape[0])
# print(predicted*0.5)
pkl.dump(predicted, open('../predicted_pkl_data/xgboost_prob_highest_result2.pkl', 'wb'), pkl.HIGHEST_PROTOCOL)