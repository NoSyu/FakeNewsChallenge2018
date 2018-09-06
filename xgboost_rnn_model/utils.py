import pickle

import numpy as np

from news.ml_model.dataset import DataSet


def get_head_body_tuples_test(data_path):
    """
        data_path에 있는 test data를 가져오는 함수

    :param
        data_path: train data의 경로
    :return:
        h : train data의 head
        b : train data의 body
    """
    dataset = DataSet(data_path, "competition_test_bodies.csv", "competition_test_stances.csv", "test_set")

    h = []
    b = []
    for stance in dataset.stances:
        h.append(stance['Headline'])
        b.append(dataset.articles[int(stance['Body ID'])])

    return h, b


def get_head_body_tuples(data_path):
    """
        data_path에 있는 train data를 가져오는 함수

    :param
        data_path: train data의 경로
    :return:
        h : train data의 head
        b : train data의 body

    """
    # file paths
    dataset = DataSet(data_path, "train_bodies.csv", "train_stances.csv", "train_set")

    h = []
    b = []
    # create new vocabulary
    for stance in dataset.stances:
        h.append(stance['Headline'])
        b.append(dataset.articles[int(stance['Body ID'])])

    print("Stances length: " + str(len(dataset.stances)))
    return h, b


def get_y_labels(data_path, one_hot=True):
    """
    y label을 가져오는 함수

    :param one_hot:
    :return:
    """
    nb_classes = 4

    if one_hot:
        train_y = pickle.load(open(data_path+'/train_y_label.pkl', 'rb'))
        train_y = np.eye(nb_classes)[np.array(train_y).reshape(-1)]
        test_y = pickle.load(open(data_path+'/test_y_label.pkl', 'rb'))
        test_y = np.eye(nb_classes)[np.array(test_y).reshape(-1)]
    else:
        train_y = pickle.load(open(data_path+'/train_y_label.pkl', 'rb'))
        test_y = pickle.load(open(data_path+'/test_y_label.pkl', 'rb'))

    return train_y, test_y
