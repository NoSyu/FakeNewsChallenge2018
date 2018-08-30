from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pkl
import numpy as np
from scipy import sparse
from scipy.sparse import hstack

class TFIDFVector_generator:

    def __init__(self, analyzer='word', ngram_range=(1, 3), stop_words='english'):
        """
        TFIDFVectorizer의 초기화하는 생성자

        :param max_features: 사용할 feature의 수
        :param analyzer: 단어의 단위를 정하는 파라미터. 'char', 'word' 옵션 사용 가능
        :param ngram_range: n-gram의 범위를 정하는 파라미터. (최소n, 최대n)의 형태로 입력
        :param stop_words: 영어의 stopword를 설정하는 파라미터. 'None', 'english'의 옵션이 선택 가능.
        :param norm: term Vector의 정규화 방식을 설정하는 파라미터. 'l1', 'l2', 'None'으로 설정 가능.
        """
        self.vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range,
                                          stop_words=stop_words)
    def fit(self, head, body, include_test = True):
        """
        TFIDFVectorizer를 학습시키는 메소드

        :param head: 학습시킬 head 문장들
        :param body: 학습시킬 body 문장들
        :return:
        """
        print('Fitting vectorizer...')
        combined_head_body = [h+". "+b for h, b in zip(head, body)]
        if include_test:
            t_head, t_body = get_head_body_tuples_test(data_path='../../data')
            combined_test_head_body = [h + ". " + b for h, b in zip(t_head, t_body)]
            combined_head_body.extend(combined_test_head_body)

        self.vectorizer.fit(combined_head_body)
        print('Fitting vectorizer finished!')
        print('vocab_len : ', len(self.vectorizer.vocabulary_))

    def save_model(self, model_path, filename):
        """
        TFIDFVectorizer 모델을 저장하는 메소드

        :param model_path: TFIDFVectorizer 모델의 저장 경로
        :param filename: TFIDFVectorizer 모델의 파일 이름
        :return:
        """
        print('Saving vectorizer...')
        pkl.dump(self.vectorizer, open(model_path+"/"+filename, 'wb'), pkl.HIGHEST_PROTOCOL)
        print('Saving vectorizer finish!')

    def load_model(self, model_path, filename):
        """
        TFIDFVectorizer 모델을 불러오는 메소드

        :param model_path: 불러올 TFIDFVectorizer 모델의 경로
        :param filename: 불러올 TFIDFVectorizer 모델 파일의 이름
        :return:
        """
        print('Loading vectorizer...')
        self.vectorizer = pkl.load(open(model_path + "/" + filename, 'rb'))
        print('Loading vectorizer finish!')

    def transform(self, head, body, with_cos=True):
        """
        학습된 TFIDFVectorizer에 문장을 넣어 TFIDF vector로 변환시키는 메소드

        :param head: 변환시킬 head 문장들
        :param body: 변환시킬 body 문장들
        :param with_cos: cos similarity를 포함할 지 여부
        :return:
            (sparse matrix 형태)
            transformed_head : TFIDF vector로 변환 된 head
            transformed_body : TFIDF vector로 변환 된 body
            combined : 변환 된 head-body pair vector
        """

        print('Transforming head...')
        transformed_head = self.vectorizer.transform(head)
        print('Transforming head finish!')
        print('Transforming body...')
        transformed_body = self.vectorizer.transform(body)
        print('Transforming body finish!')

        print('Combining features...')
        combined = np.concatenate((transformed_head.toarray(), transformed_body.toarray()), axis=1)
        combined = sparse.csr_matrix(combined)
        print('Combining features finish!')

        return transformed_head, transformed_body, combined

    def transform_and_save_data(self, head, body, save_path, filename):
        """
        학습된 TFIDFVectorizer에 문장을 넣어 TFIDF vector로 변환 및 저장하는 메소드

        :param head: 변환시킬 head 문장들
        :param body: 변환시킬 body 문장들
        :param save_path: 저장할 경로
        :param filename: 저장할 파일 이름
        :param with_cos: cos similarity를 포함할 지 여부
        :return:
        """
        print('Transforming head...')
        transformed_head = self.vectorizer.transform(head)
        print('Transforming head finish!')
        print('Transforming body...')
        transformed_body = self.vectorizer.transform(body)
        print('Transforming body finish!')

        print('Combining features...')
        combined = hstack((transformed_head, transformed_body), format='csr')
        print('Combining features finish!')

        head_file_name = save_path+"/"+filename+"_head.pkl"
        body_file_name = save_path+"/"+filename+"_body.pkl"
        combined_file_name = save_path+"/"+filename+"_combined.pkl"

        print('Saving transformed TFIDF head body vectors...')
        pkl.dump(transformed_head, open(head_file_name, 'wb'), pkl.HIGHEST_PROTOCOL)
        pkl.dump(transformed_body, open(body_file_name, 'wb'), pkl.HIGHEST_PROTOCOL)
        pkl.dump(combined, open(combined_file_name, 'wb'), pkl.HIGHEST_PROTOCOL)
        print('Saving finish TFIDF head body vectors')
        print('Head : {}\nBody : {}\nCombined : {}\n'
              .format(head_file_name, body_file_name, combined_file_name))

if __name__ == "__main__":
    from Tree_models.utils.get_input_datas import get_head_body_tuples, get_head_body_tuples_test
    head, body = get_head_body_tuples(data_path='../../data')
    head_test, body_test = get_head_body_tuples_test(data_path='../../data')

    model_path = '../../pickled_data'
    max_features = 200000
    # filename = 'tfidf_'+str(max_features)+'_vecterizer_model.pkl'
    filename = 'tfidf_1st_vecterizer_model.pkl'

    tfidf = TFIDFVector_generator(analyzer='word', ngram_range=(1, 3), stop_words='english')
    tfidf.fit(head, body)
    tfidf.save_model(model_path=model_path, filename=filename)

    # 저장된 Vecterizer 모델이 있으면 load_model을 사용하면 됨
    # tfidf.load_model(model_path=model_path, filename=filename)

    # 저장된 TFIDF vector가 있으면 바로 해당 데이터로 training 하면 됨
    # tfidf.transform_and_save_data(head, body, save_path=model_path,
    #                               filename='tfidf_feat_'+str(max_features))
    #
    # tfidf.transform_and_save_data(head_test, body_test, save_path=model_path,
    #                               filename='tfidf_feat_'+str(max_features))
    tfidf.transform_and_save_data(head, body, save_path=model_path,
                                  filename='tfidf_feat_1st_train_include_test')

    tfidf.transform_and_save_data(head_test, body_test, save_path=model_path,
                                  filename='tfidf_feat_1st_test_include_test')
    # cos similarity 모델을 만들기 위해서는 메모리가 터지지 않게
    # CosineSim_generator의 함수들을 이용해 head, body pkl 파일을 읽어서 만들면 됩니다.


