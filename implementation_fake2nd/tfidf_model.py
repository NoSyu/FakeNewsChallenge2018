from implementation.implement_model_function_train_test_tfidf_5000 import MLP_Classifier
import os

base_dir = os.path.dirname(__file__)
row_body_train = base_dir+'/data/train_bodies.csv'
row_stance_train = base_dir+'/data/train_stances.csv'
head_dir_train = base_dir+'/pickled_model/tfidf_head_feature_train.pkl'
body_dir_train = base_dir+'/pickled_model/tfidf_body_feature_train.pkl'
label_dir_train = base_dir+'/pickled_model/tfidf_label_one_hot_train.pkl'

row_body_test = base_dir+'/data/competition_test_bodies.csv'
row_stance_test = base_dir+'/data/competition_test_stances.csv'
head_dir_test = base_dir+'/pickled_model/tfidf_head_feature_test.pkl'
body_dir_test = base_dir+'/pickled_model/tfidf_body_feature_test.pkl'
label_dir_test = base_dir+'/pickled_model/tfidf_label_one_hot_test.pkl'

save_model_path = base_dir+'/tf_model/tfidf_5000_epoch'

try:
    mode = input('사용할 mode를 입력하세요. test 또는 train (default : train) => ').strip().lower()
    if mode == 'train':
        MLP_Classifier(row_body_train, row_stance_train, row_body_test, row_stance_test,
                           head_dir_train, body_dir_train, label_dir_train,
                           head_dir_test, body_dir_test, label_dir_test,
                           learning_rate=0.001, batch_size=188,
                           training_epoch=70,
                           init_bias=0.001, mode='train', save_model_path=save_model_path)

    elif mode == 'test':
        MLP_Classifier(row_body_train, row_stance_train, row_body_test, row_stance_test,
                           head_dir_train, body_dir_train, label_dir_train,
                           head_dir_test, body_dir_test, label_dir_test,
                           learning_rate=0.001, batch_size=188,
                           training_epoch=70,
                           init_bias=0.001, mode='test', save_model_path=save_model_path)

except:
    print('입력 오류입니다.')
    print('프로그램을 종료합니다.')
    exit(-1)