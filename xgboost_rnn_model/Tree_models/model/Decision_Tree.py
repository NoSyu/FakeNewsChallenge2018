from sklearn.tree import DecisionTreeClassifier
import pickle as pkl
from Tree_models.utils.score import report_score
import numpy as np

class Decision_Tree:

    def __init__(self):
        """
        Decision Tree 초기화하는 생성자
        
        """
        self.tree = DecisionTreeClassifier()

    def fit(self, X, y):
        """
        Decision Tree를 학습시키는 메소드
        
        :param X: input data 형태 [[0, 0], [0, 1], [1, 0], [1, 1]] 
        :param y: label data 형태 [[1, 0], [0, 1], [0, 1], [1, 0]]
        :return: 
        """
        print('Decision tree training...')
        self.tree.fit(X, y)
        print('Decision tree train finish!')

    def save_model(self, save_file_path, model_name):
        """
        학습한 Decision Tree를 pkl 파일로 저장하는 메소드
        
        :param save_file_path: 모델 저장 경로 
        :param model_name: 모델 이름
        :return: 
        """
        full_path = save_file_path + "/" + model_name

        print('Decision tree {} file saving...'.format(model_name))
        pkl.dump(self.tree, open(full_path, 'wb'), pkl.HIGHEST_PROTOCOL)
        print('File saving finish!')

    def load_model(self, load_file_path, model_name):
        """
        pkl 파일로 저장된 Decision Tree를 불러오는 메소드
        
        :param load_file_path: 불러올 모델 경로 
        :param model_name: 모델 이름
        :return: 
        """
        full_path = load_file_path + "/" + model_name

        print('Load decision tree path : {}'.format(full_path))
        self.tree = pkl.load(open(full_path, 'rb'))
        print('Load finish!')

    def predict_and_scoring(self, test_X, test_y):
        # print(test_X[:10])
        """
        학습한 Tree 모델을 이용해 예측 및 fnc 스코어로 채점하는 메소드
    
        :param test_X: test input data 형태 [[0, 0], [0, 1], [1, 0], [1, 1]]
        :param test_y: test label data 형태 [[1, 0], [0, 1], [0, 1], [1, 0]]
        :return:
        """
        print('Start prediction!')

        predicted = self.tree.predict(test_X)
        # print(predicted[:10], test_y[:10])
        # print(np.argmax(predicted[:10]), test_y)
        LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
        LABELS_RELATED = ['unrelated', 'related']
        RELATED = LABELS[0:3]

        if type(predicted[0]) == np.int32:
            print('predicted datas are label number')
            report_score([LABELS[e] for e in np.argmax(test_y, axis=1)],
                         [LABELS[e] for e in predicted])
        else:
            print('predicted datas are one hot')
            report_score([LABELS[e] for e in np.argmax(test_y, axis=1)],
                         [LABELS[e] for e in np.argmax(predicted, axis=1)])