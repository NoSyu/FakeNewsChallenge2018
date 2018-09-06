from Tree_models.utils.get_input_datas import get_head_body_tuples, get_head_body_tuples_test
from Tree_models.utils.get_input_datas import get_y_labels
from Tree_models.utils.score import report_score
import pickle as pkl
from keras.models import load_model
import numpy as np
from scipy.sparse import hstack
import xgboost as xgb
from sklearn.preprocessing import normalize

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

def load_GRU_new_model(model_path):
    return load_model(model_path+"/stacked_GRU_4dense.h5")
    # return load_model(model_path+"/GRU_save.h5")

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

def load_stacking_logreg_model():
    return load_pkl('./pickled_data', 'stacking_logreg_model.pkl')

def get_train_X(data_path):
    train_tf_include_holdout_name = 'count_1st_train_include_test_combined.pkl'
    train_tfidf_include_holdout_name = 'tfidf_cos_train_include_holdout.pkl'
    train_nmf_name = 'nmf_200_cos_train_include_holdout.pkl'
    train_svd_name = 'svd_100_cos_train_include_holdout.pkl'

    train_tf_X = load_pkl(data_path, train_tf_include_holdout_name)
    train_tfidf_X = load_pkl(data_path, train_tfidf_include_holdout_name)
    train_svd_X = load_pkl(data_path, train_svd_name)
    train_nmf_X = load_pkl(data_path, train_nmf_name)

    train_X = normalize(hstack((train_tf_X, train_tfidf_X,
                                train_svd_X, train_nmf_X), format='csr'))
    return train_X

def get_test_X(data_path):
    test_tf_include_holdout_name = 'count_1st_test_include_test_combined.pkl'
    test_tfidf_include_holdout_name = 'tfidf_cos_test_include_holdout.pkl'
    test_nmf_name = 'nmf_200_cos_test_include_holdout.pkl'
    test_svd_name = 'svd_100_cos_test_include_holdout.pkl'

    test_tf_X = load_pkl(data_path, test_tf_include_holdout_name)
    test_tfidf_X = load_pkl(data_path, test_tfidf_include_holdout_name)
    test_svd_X = load_pkl(data_path, test_svd_name)
    test_nmf_X = load_pkl(data_path, test_nmf_name)

    test_X = normalize(hstack((test_tf_X, test_tfidf_X,
                               test_svd_X, test_nmf_X), format='csr'))
    # print(test_tf_X.shape)
    # print(test_tfidf_X.shape)
    # print(test_svd_X.shape)
    # print(test_nmf_X.shape)

    return test_X

def get_test_X_gru(data_path):
    X_test_GRU = pkl.load(open(data_path+'/sequences_test.pkl', 'rb'))
    return X_test_GRU

def print_intro():
    print('#######################################')
    print('##         Fake news detector        ##')
    print('#######################################')
    print('1. Index를 이용한 파일 단위 detection')
    print('2. Test 전체 결과 출력')
    print('#######################################\n')


def main():
    datapath = './data'
    pickled_path = './pickled_data'
    rnn_feat_path = './rnn_features'

    gru_model_path = './saved_keras'
    xgb_model_path = './xgboost_model_saved_file'

    xgb_params = load_XGB_params()

    xgb_model = load_XGB_model(xgb_model_path, xgb_params)
    gru_model = load_GRU_new_model(gru_model_path)
    stacking_log_model = load_stacking_logreg_model()

    test_dataset = read_dataset_test_head_body(datapath)
    test_features_xgb = get_test_X(pickled_path)
    test_features_gru = get_test_X_gru(rnn_feat_path)
    # print([test_features_gru[0]])
    _, test_y = get_y_labels(pickled_path, one_hot=False)
    # print(test_y)
    LABELS = ['agree', 'disagree', 'discuss', 'unrelated']

    while True:
        print_intro()

        try:
            num = input('번호를 입력하세요. 나가기(Q or q) : ')

        except Exception as e:
            print(e)
            print('잘못된 입력입니다.')
            continue

        if num.lower() == 'q':
            print('프로그램을 종료합니다.')
            break
        num = int(num)
        if num == 1:
            try:
                index = int(input('파일 Index 번호를 입력하세요 : '))
            except Exception as e:
                print(e)
                print('입력 오류입니다.')
                break
            feature = xgb.DMatrix(test_features_xgb.getrow(index))

            data = test_dataset[index]
            print('#############################################################\n')
            print('head : {}\n'.format(data['head']))
            print('#############################################################\n')
            print('body : {}\n'.format(data['body']))
            print('#############################################################\n')

            xgb_pred = xgb_model.predict(feature)
            gru_pred = gru_model.predict(np.array([test_features_gru[index]]))
            input_data = np.hstack((xgb_pred, gru_pred))

            pred = stacking_log_model.predict_proba(input_data)
            argmax = np.argmax(pred, axis=1)[0]

            if argmax != test_y[index]:
                print('오분류')

            print('예측 결과 : ', LABELS[argmax], ', 실제 정답 : ', LABELS[test_y[index]])
            pred = pred[0]
            print('agree : {:f}, disagree : {:f}, discuss : {:f}, unrelated : {:f}\n'.
                      format(pred[0], pred[1], pred[2], pred[3]))
            print('#############################################################\n\n')
            
        elif num == 2:
            feature = xgb.DMatrix(test_features_xgb, label=test_y)

            xgb_pred = xgb_model.predict(feature)
            gru_pred = gru_model.predict(test_features_gru)
            input_data = np.hstack((xgb_pred, gru_pred))

            pred = stacking_log_model.predict_proba(input_data)

            pred = np.argmax(pred, axis=1)

            report_score([LABELS[int(e)] for e in test_y], [LABELS[int(e)] for e in np.argmax(xgb_pred, axis=1)])
            report_score([LABELS[int(e)] for e in test_y], [LABELS[int(e)] for e in np.argmax(gru_pred, axis=1)])
            report_score([LABELS[int(e)] for e in test_y], [LABELS[int(e)] for e in pred])
        print()

if __name__ == '__main__':
    main()