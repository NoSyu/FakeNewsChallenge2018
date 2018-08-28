
import numpy as np
import pickle

from keras_model.feature_helper.extract_y_feature import load_y_data
from keras_model.models.single_f_ext_LSTM import single_f_ext_GRU

if __name__ == "__main__":

    gru = single_f_ext_GRU(save_folder='./saved_keras/')
    X_train_GRU = pickle.load(open('./rnn_features/sequences_train.pkl', 'rb'))
    # X_train_MLP = pickle.load(open('/rnn_features/tfidf_head_body_200_train.pkl', 'rb'))
    y_train = np.array(load_y_data('./rnn_features', 'train_y_label.pkl'))
    X_test_GRU = pickle.load(open('./rnn_features/sequences_test.pkl', 'rb'))
    # X_test_MLP = pickle.load(open('keras_model/rnn_features/tfidf_head_body_200_test.pkl', 'rb'))
    y_test = np.array(load_y_data('./rnn_features', 'test_y_label.pkl'))
    loss_filename = 'single_gru_lossfile'
    #
    X_train = X_train_GRU
    X_test = X_test_GRU
    #
    #
    gru.fit(X_train, y_train, X_test, y_test, loss_filename)

    from keras_model.utils.score import report_score
    LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
    LABELS_RELATED = ['unrelated', 'related']
    RELATED = LABELS[0:3]

    pred = gru.predict(X_test_GRU)
    report_score([LABELS[e] for e in y_test], [LABELS[e] for e in pred])
    pred_prob = gru.predict_prob(X_test_GRU)
    print(pred_prob[:100])
    pickle.dump(pred_prob, open('./predicted_pkl_data/single_GRU_prob_result.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

     # print(pred