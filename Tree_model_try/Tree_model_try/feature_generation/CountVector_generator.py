from sklearn.feature_extraction.text import CountVectorizer
import pickle as pkl
import numpy as np
from scipy import sparse
class CountVector_generator:

    def __init__(self, max_features=2500, analyzer='word', ngram_range=(1, 1), stop_words='english'):
        """
        CountVectorizer의 초기화하는 생성자

        :param max_features: 사용할 feature의 수
        :param analyzer: 단어의 단위를 정하는 파라미터. 'char', 'word', 'char_wb' 옵션 사용 가능
        :param ngram_range: n-gram의 범위를 정하는 파라미터. (최소n, 최대n)의 형태로 입력
        :param stop_words: 영어의 stopword를 설정하는 파라미터. 'None', 'english'의 옵션이 선택 가능.
        """
        self.max_featrues = max_features
        self.vectorizer = CountVectorizer(max_features=max_features, analyzer=analyzer, ngram_range=ngram_range,
                                          stop_words=stop_words)
    def fit(self, head, body):
        """
        CountVectorizer를 학습시키는 메소드

        :param head: 학습시킬 head 문장들
        :param body: 학습시킬 body 문장들
        :return:
        """
        print('Fitting vectorizer...')
        combined_head_body = [h + ". " + b for h, b in zip(head, body)]
        self.vectorizer.fit(combined_head_body)
        print('Fitting vectorizer finished!')

    def save_model(self, model_path, filename):
        """
        CountVectorizer 모델을 저장하는 메소드

        :param model_path: CountVectorizer 모델의 저장 경로
        :param filename: CountVectorizer 모델의 파일 이름
        :return:
        """
        print('Saving vectorizer...')
        pkl.dump(self.vectorizer, open(model_path + "/" + filename, 'wb'), pkl.HIGHEST_PROTOCOL)
        print('Saving vectorizer finish!')

    def load_model(self, model_path, filename):
        """
        CountVectorizer 모델을 불러오는 메소드

        :param model_path: 불러올 CountVectorizer 모델의 경로
        :param filename: 불러올 CountVectorizer 모델 파일의 이름
        :return:
        """
        print('Loading vectorizer...')
        self.vectorizer = pkl.load(open(model_path + "/" + filename, 'rb'))
        print('Loading vectorizer finish!')

    def transform(self, head, body):
        """
        학습된 CountVectorizer에 문장을 넣어 Count vector로 변환시키는 메소드

        :param head: 변환시킬 head 문장들
        :param body: 변환시킬 body 문장들
        :return:
            (sparse matrix 형태)
            transformed_head : Count vector로 변환 된 head
            transformed_body : Count vector로 변환 된 body
            combined : 변환 된 head-body pair vector
        """
        print('Transforming head...')
        transformed_head = self.vectorizer.transform(head)
        print('Transforming head finish!')
        print('Transforming body...')
        transformed_body = self.vectorizer.transform(body)
        print('Transforming body finish!')
        print('Combining features...')
        combined = sparse.csr_matrix(
            np.concatenate((transformed_head.toarray(), transformed_body.toarray()), axis=1))
        print('Combining features finish!')

        return transformed_head, transformed_body, combined

    def transform_and_save_data(self, head, body, save_path, filename):
        """
        학습된 CountVectorizer에 문장을 넣어 Count vector로 변환 및 저장하는 메소드

        :param head: 변환시킬 head 문장들
        :param body: 변환시킬 body 문장들
        :param save_path: 저장할 경로
        :param filename: 저장할 파일 이름
        :return:
        """
        print('Transforming head...')
        transformed_head = self.vectorizer.transform(head)
        print('Transforming head finish!')
        print('Transforming body...')
        transformed_body = self.vectorizer.transform(body)
        print('Transforming body finish!')
        combined = sparse.csr_matrix(
            np.concatenate((transformed_head.toarray(), transformed_body.toarray()), axis=1))

        head_file_name = save_path+"/"+filename+"_head.pkl"
        body_file_name = save_path+"/"+filename+"_body.pkl"
        combined_file_name = save_path+"/"+filename+"_combined.pkl"

        print('Saving transformed Count head body vectors...')
        pkl.dump(transformed_head, open(head_file_name, 'wb'), pkl.HIGHEST_PROTOCOL)
        pkl.dump(transformed_body, open(body_file_name, 'wb'), pkl.HIGHEST_PROTOCOL)
        pkl.dump(combined, open(combined_file_name, 'wb'), pkl.HIGHEST_PROTOCOL)
        print('Saving finish Count head body vectors')
        print('Head : {}\nBody :{}\nCombined : {}\n'
              .format(head_file_name, body_file_name, combined_file_name))

if __name__ == "__main__":
    from Tree_model_try.utils.get_input_datas import get_head_body_tuples, get_head_body_tuples_test
    model_path = '../../pickled_data'

    max_features = 5000 # 메모리가 터질시 max_features를 낮게 조정

    filename = 'count_'+str(max_features)+'_vecterizer_model.pkl'

    head, body = get_head_body_tuples(data_path='../../data')
    head_test, body_test = get_head_body_tuples_test(data_path='../../data')

    count_vectorizer = CountVector_generator\
        (max_features=max_features, analyzer='word', ngram_range=(1, 1), stop_words='english')

    # count_vectorizer.fit(head, body)
    # 저장된 Vecterizer 모델이 있으면 load_model을 사용하면 됨
    # count_vectorizer.save_model(model_path=model_path, filename=filename)
    count_vectorizer.load_model(model_path=model_path, filename=filename)

    # 저장된 TFIDF vector가 있으면 바로 해당 데이터로 training 하면 됨
    # 변환된 train file 저장
    count_vectorizer.transform_and_save_data(head, body, save_path=model_path, filename='count_' + str(max_features))
    # 변환된 test file 저장
    count_vectorizer.transform_and_save_data(head_test, body_test, save_path=model_path, filename='count_' + str(max_features)+'_test')