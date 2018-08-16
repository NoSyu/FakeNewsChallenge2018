from sklearn.feature_extraction.text import TfidfVectorizer
from implementation.model_control import save_model
from implementation.model_control import load_model
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from data_helper.Dataset_new import Dataset
import os
from implementation.model_control import load_model, save_model

class Feature_enginnering():

    def __init__(self, body, head, stance):
        """
        Dataset_new.py에서 받아온 데이터들을 이용해 feature engineering을 하는 클래스 생성자
        :param body: 전처리 된 body 문장들.(head-body-stance의 순서가 같아야 함)
        :param head: 전처리 된 head 문장들.
        :param stance: one-hot vector로 표현된 y_label
        """
        self.body = body
        self.head = head
        self.stance = stance
        self.save_path = '../pickled_model'

    def set_save_path(self, new_path):
        """
        feature 파일들을 저장할 위치를 설정하는 메소드
        :param new_path: 새로운 파일 저장 경로
        """
        self.save_path = new_path
        print('Change feature save path : ', new_path)

    def get_tfidf_vocab_5000(self):
        """
        TF-IDF 벡터를 만들기 위한 train_vocab 파일을 반환 하는 메소드
        :return: train용 TF-IDF vocab 파일 
        """
        train_data = [b + " " + h for b, h in zip(self.body, self.head)]

        model = TfidfVectorizer(max_features=5000, ngram_range=(1, 1),
                                stop_words='english',
                                norm='l2', use_idf=False)
        model.fit_transform(train_data)

        self.vocab = model.vocabulary_
        return self.vocab

    def get_tfidf_vocab_5000_holdout(self, test_body, test_stance):
        """
        TF-IDF 벡터를 만들기 위한 train_vocab 파일을 반환 하는 메소드
        :return: train용 TF-IDF vocab 파일
        """
        test_dataset = Dataset(test_body, test_stance)
        t_h, t_b = test_dataset.read_tfidf_data()
        test_h = [h for h in t_h]
        test_b = [b for b in t_b]
        train_data = [b + " " + h for b, h in zip(self.body, self.head)]
        train_data.extend(test_b)
        train_data.extend(test_h)

        model = TfidfVectorizer(max_features=5000, ngram_range=(1, 1),
                                stop_words='english',
                                norm='l2', use_idf=False)
        model.fit_transform(train_data)
        if os.path.exists('../pickled_model/tfidf_holdout_vocab.pkl'):
            self.vocab = load_model('../pickled_model/tfidf_holdout_vocab.pkl')
            print('vocab loaded!')
        else:
            self.vocab = model.vocabulary_
            save_model('../pickled_model/tfidf_holdout_vocab.pkl', model.vocabulary_)
            return self.vocab

    def get_tfidf_vocab_100(self):
        """
        TF-IDF 벡터를 만들기 위한 train_vocab 파일을 반환 하는 메소드
        :return: train용 TF-IDF vocab 파일
        """

        train_data = [b + " " + h for b, h in zip(self.body, self.head)]

        model = TfidfVectorizer(max_features=100, ngram_range=(1, 1),
                                stop_words='english',
                                norm='l2', use_idf=False)
        model.fit_transform(train_data)

        self.vocab = model.vocabulary_

        return self.vocab

    def get_tfidf_vocab_100_holdout(self, test_body, test_stance):
        """
        TF-IDF 벡터를 만들기 위한 train_vocab 파일을 반환 하는 메소드
        :return: train용 TF-IDF vocab 파일
        """
        test_dataset = Dataset(test_body, test_stance)
        t_h, t_b = test_dataset.read_tfidf_data()
        test_h = [h for h in t_h]
        test_b = [b for b in t_b]
        train_data = [b + " " + h for b, h in zip(self.body, self.head)]
        train_data.extend(test_b)
        train_data.extend(test_h)

        model = TfidfVectorizer(max_features=100, ngram_range=(1, 1),
                                stop_words='english',
                                norm='l2', use_idf=False)
        model.fit_transform(train_data)

        self.vocab = model.vocabulary_
        return self.vocab
    def tfidf_train_body(self, filename, model_save=False):
        """
        train body 데이터를 TF-IDF 벡터로 만들어주는 메소드
        :param filename: body 파일이 존재하는 경로
        :param model_save: 모델 저장 여부
        :return: 만들어진 body 모델
        """
        model = TfidfVectorizer(vocabulary=self.vocab, use_idf=True,
                                norm="l2", stop_words='english')

        X_body = model.fit_transform(self.body)

        if model_save:
            saved_path = self.save_path+"/"+filename
            print('tfidf_body_feature saving......')
            save_model(saved_path, X_body)
            print('feature saving finished!')
            print('saved path : ', saved_path)
        else:
            return X_body

    def tfidf_train_head(self, filename, model_save=False):

        model = TfidfVectorizer(vocabulary=self.vocab, use_idf=True,
                                norm="l2", stop_words='english')

        X_head = model.fit_transform(self.head)
        if model_save:
            saved_path = self.save_path+"/"+filename
            print('tfidf_head_feature saving......')
            save_model(saved_path, X_head)
            print('feature saving finished!')
            print('saved path : ', saved_path)
        else:
            return X_head

    def tfidf_stance_save(self, filename, model_save=False):
        if model_save:
            saved_path = self.save_path + "/"+ filename
            print('tfidf_stance_one_hot saving......')
            save_model(saved_path, self.stance)
            print('feature saving finished!')
            print('saved path : ', saved_path)
        else:
            return self.stance

    def tfidf_cos_save(self, head_path, body_path, filename, model_save=False):
        head = load_model(head_path).toarray()
        body = load_model(body_path).toarray()
        cos = []
        for x, y in zip(head, body):
            x = x.reshape(1, -1)
            y = y.reshape(1, -1)
            value = cosine_similarity(x, y)[0]
            cos.append(value)
        cos = np.array(cos)

        if model_save:
            saved_path = self.save_path + "/" + filename
            print('tfidf_cos saving......')
            save_model(saved_path, cos)
            print('feature saving finished!')
            print('saved path : ', saved_path)
        else:
            return cos

