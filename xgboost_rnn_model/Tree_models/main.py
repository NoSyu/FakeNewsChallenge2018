from Tree_models.utils.get_input_datas import get_y_labels
from Tree_models.model.Decision_Tree import Decision_Tree
import pickle as pkl
import xgboost as xgb
from Tree_models.utils.score import report_score

def load_pkl(file_path, filename):
    print('Loading {}/{}'.format(file_path, filename))
    pkl_file = pkl.load(open(file_path + "/" + filename, 'rb'))
    print('Load {}/{} finish!'.format(file_path, filename))
    return pkl_file

model = 'xgboost'
file_path = '../pickled_data'
model_path = '../saved_model'

if model == 'tfidf_cos_5000':

    train_X_count = 'tfidf_5000_cosine.pkl'
    train_y_filename = 'train_y_label.pkl'
    test_X_filename = 'tfidf_5000_test_cosine.pkl'
    test_y_filename = 'test_y_label.pkl'

    train_X = load_pkl(file_path, train_X_count).toarray()
    test_X = load_pkl(file_path, test_X_filename).toarray()
    train_y, test_y = get_y_labels(file_path, one_hot=True)

    clf = Decision_Tree()
    # clf.fit(train_X, train_y)
    # clf.save_model(save_file_path=model_path,
    #                model_name='decision_tree_5000_cosine.pkl')
    clf.load_model(load_file_path=model_path, model_name='decision_tree_5000_cosine.pkl')
    clf.predict_and_scoring(test_X, test_y)

elif model == 'count_5000':
    train_X_count = 'count_5000_combined.pkl'
    train_y_filename = 'train_y_label.pkl'
    test_X_filename = 'count_5000_test_combined.pkl'
    test_y_filename = 'test_y_label.pkl'

    train_X = load_pkl(file_path, train_X_count).toarray()
    test_X = load_pkl(file_path, test_X_filename).toarray()
    train_y, test_y = get_y_labels(file_path, one_hot=True)

    clf = Decision_Tree()
    clf.fit(train_X, train_y)
    clf.save_model(save_file_path=model_path,
                   model_name='decision_tree_count_5000.pkl')
    clf.predict_and_scoring(test_X, test_y)

elif model =='count_cos_5000':
    train_X_count = 'count_5000_cosine.pkl'
    train_y_filename = 'train_y_label.pkl'
    test_X_filename = 'count_5000_test_cosine.pkl'
    test_y_filename = 'test_y_label.pkl'

    train_X = load_pkl(file_path, train_X_count).toarray()
    test_X = load_pkl(file_path, test_X_filename).toarray()
    train_y, test_y = get_y_labels(file_path, one_hot=True)

    clf = Decision_Tree()
    clf.fit(train_X, train_y)
    clf.save_model(save_file_path=model_path,
                   model_name='decision_tree_count_cosine_5000.pkl')
    clf.predict_and_scoring(test_X, test_y)

elif model == 'naive_tfidf_5000':
    from sklearn.naive_bayes import MultinomialNB
    import numpy as np
    train_X_count = 'tfidf_5000_combined.pkl'
    train_y_filename = 'train_y_label.pkl'
    test_X_filename = 'tfidf_5000_test_combined.pkl'
    test_y_filename = 'test_y_label.pkl'

    train_X = load_pkl(file_path, train_X_count).toarray()
    test_X = load_pkl(file_path, test_X_filename).toarray()
    train_y, test_y = get_y_labels(file_path, one_hot=True)
    # print(train_X[0])
    # print(np.array(train_y).reshape(1, -1))
    clf = MultinomialNB()
    clf.fit(train_X, train_y)
    predicted = clf.predict(test_X)
    LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
    LABELS_RELATED = ['unrelated', 'related']
    RELATED = LABELS[0:3]
    report_score([LABELS[e] for e in test_y], [LABELS[e] for e in predicted])
elif model == 'xgboost':
    from scipy import sparse
    from scipy.sparse import hstack
    import numpy as np


    # 'objective': 'multi:softmax',
    param = {
        'max_depth' : 6,#(defalut 6)
        'colsample_bytree' : 0.6,
        'subsample' : 1.0, # 트리마다의 관측 데이터 샘플링 비율, 값이 낮아질수록 under-fitting(default=1)
        'eta' : 0.01,
        'objective' : 'multi:softprob',
        'num_class' : 4,
        'nthread' : 14,
        'silent' : 1,
        # 'gpu_id' : 0,
        # 'max_bin': 16,
        # 'tree_method' : 'gpu_hist',
        'eval_metric' : 'mlogloss',
        'seed' : 2017
    }

    num_round = 1000

    # train_count_name = 'count_5000_combined.pkl'
    # train_tfidf_name = 'tfidf_5000_cosine.pkl'
    # train_nmf_name = 'nmf_200_cosine.pkl'
    # train_y_name = 'train_y_label.pkl'
    #
    # test_count_name = 'count_5000_test_combined.pkl'
    # test_tfidf_name = 'tfidf_5000_test_cosine.pkl'
    # test_nmf_name = 'nmf_200_test_cosine.pkl'
    # test_y_name = 'test_y_label.pkl'

    # train_count_name = ''
    train_tf_name = 'count_1st_train_combined.pkl'
    train_tfidf_name = 'tfidf_feat_1st_cos_train.pkl'
    train_nmf_name = 'nmf_feat_200_cos_train.pkl'
    train_svd_name = 'svd_feat_50_cos_train.pkl'
    # train_sent_name = 'senti_train_cos.pkl'
    train_y_name = 'train_y_label.pkl'
    # test_count_name = ''

    test_tf_name = 'count_1st_test_combined.pkl'
    test_tfidf_name = 'tfidf_feat_1st_cos_test.pkl'
    test_nmf_name = 'nmf_feat_200_cos_test.pkl'
    test_svd_name = 'svd_feat_50_cos_test.pkl'
    # test_sent_name = 'senti_test_cos.pkl'
    test_y_name = 'test_y_label.pkl'

    saved_path = '../saved_model'
    pickled_path = '../pickled_data'

    train_tf_X = load_pkl(pickled_path, train_tf_name)
    train_tfidf_X = load_pkl(saved_path, train_tfidf_name)
    train_svd_X = load_pkl(saved_path, train_svd_name)
    train_nmf_X = load_pkl(saved_path, train_nmf_name)
    # train_sentiment_X = load_pkl(pickled_path, train_sent_name)

    train_X = sparse.hstack((train_tf_X, train_tfidf_X, train_nmf_X, train_svd_X), format='csr')
    # exit()

    test_tf_X = load_pkl(pickled_path, test_tf_name)
    test_tfidf_X = load_pkl(saved_path, test_tfidf_name)
    test_svd_X = load_pkl(saved_path, test_svd_name)
    test_nmf_X = load_pkl(saved_path, test_nmf_name)
    # test_sentiment_X = load_pkl(pickled_path, test_sent_name)

    test_X = sparse.hstack((test_tf_X, test_tfidf_X, test_nmf_X, test_svd_X), format='csr')

    train_y, test_y = get_y_labels(file_path, one_hot=False)
    # print(train_y)
    # print(test_y)
    # print(test_y[:2])
    # print(np.argmax(test_y[:2], axis=1))
    # exit()
    # print(train_y[:2])
    dtrain = xgb.DMatrix(train_X, label=train_y)
    dtest = xgb.DMatrix(test_X, label=test_y)
    # exit()
    watch_list = [(dtrain, 'train'), (dtest, 'test')]
    bst = xgb.train(param, dtrain, num_round, watch_list, verbose_eval=10, early_stopping_rounds=50)
    model_name = '../saved_model/tf_tfidf_cos_svd_nmf_epoch_'+str(num_round)+'_xgboost_softprob_1st.model'
    bst.save_model(model_name)

    # if load models
    # bst = xgb.Booster()
    # bst.load_model(model_name)
    # predicted = bst.predict(dtest)
    predicted = bst.predict(dtest).reshape(test_X.shape[0], 4)

    LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
    LABELS_RELATED = ['unrelated', 'related']
    RELATED = LABELS[0:3]
    # print(predicted[:3])
    # print(np.argmax(pred icted, axis=1))
    # report_score([LABELS[int(e)] for e in test_y], [LABELS[int(e)] for e in predicted])
    report_score([LABELS[int(e)] for e in test_y], [LABELS[int(e)] for e in np.argmax(predicted, axis=1)])
