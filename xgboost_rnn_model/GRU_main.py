
import numpy as np
import pickle

from keras_model.feature_helper.extract_y_feature import load_y_data
from keras_model.models.single_f_ext_LSTM import single_f_ext_GRU
from keras_model.utils.score import report_score

if __name__ == "__main__":
    mode = 'test'

    ####
    # 사용할 gru 객체를 불러옴
    ####
    gru = single_f_ext_GRU(save_folder='./saved_keras/')

    ####
    #   GRU 입력 데이터를 읽어오는 부분
    ####
    X_train = pickle.load(open('./rnn_features/sequences_train.pkl', 'rb'))
    # X_train_MLP = pickle.load(open('/rnn_features/tfidf_head_body_200_train.pkl', 'rb'))
    y_train = np.array(load_y_data('./rnn_features', 'train_y_label.pkl'))
    X_test = pickle.load(open('./rnn_features/sequences_test.pkl', 'rb'))
    # X_test_MLP = pickle.load(open('keras_model/rnn_features/tfidf_head_body_200_test.pkl', 'rb'))
    y_test = np.array(load_y_data('./rnn_features', 'test_y_label.pkl'))
    loss_filename = 'single_gru_lossfile'
    #


    if mode == 'train':
        ####
        #   GRU 모델을 학습시키는 부분
        ####
        gru.fit(X_train, y_train, X_test, y_test, loss_filename)
        
    if mode == 'test':
        LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
        LABELS_RELATED = ['unrelated', 'related']
        RELATED = LABELS[0:3]
        
        ####
        #   학습된 GRU 모델으로 예측하는 부분
        ####
        pred = gru.predict(X_test)
        
        # pred = gru.predict(X_train)
        report_score([LABELS[e] for e in y_train], [LABELS[e] for e in pred])
        
        # pred_prob = gru.predict_prob(X_test)
        # pred_prob = gru.predict_prob(X_train)
        # print(pred_prob[:100])
        # pickle.dump(pred_prob, open('./predicted_pkl_data/stacked_GRU_prob_result_train.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

     # print(pred
