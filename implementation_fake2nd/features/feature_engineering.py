from sklearn.feature_extraction.text import TfidfVectorizer
from implementation.model_control import save_model

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

    def get_tfidf_vocab(self):
        """
        TF-IDF 벡터를 만들기 위한 train_vocab 파일을 반환 하는 메소드
        :return: train용 TF-IDF vocab 파일 
        """
        all_data = [b + " " + h for b, h in zip(self.body, self.head)]

        model = TfidfVectorizer(max_features=5000, ngram_range=(1, 1),
                                stop_words='english',
                                norm='l2', use_idf=False)
        model.fit_transform(all_data)

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

