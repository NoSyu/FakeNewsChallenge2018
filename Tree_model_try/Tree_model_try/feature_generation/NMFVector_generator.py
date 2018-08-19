from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
import pickle as pkl
import numpy as np

class NMF_generator:

    def __init__(self, n_components=200, init='nndsvd', solver='mu'):
        """
        NMF 모델을 초기화하는 생성자

        """
        self.NMF = NMF(n_components=n_components, init=init, solver=solver)

    def fit(self, head_idf, body_idf):
        """
        NMF를 수행하는 메소드

        :param head: 학습시킬 head tfidf 벡터들
        :param body: 학습시킬 body tfidf 벡터들
        :return:
        """
        self.NMF.fit(np.vstack([head_idf, body_idf]))

    def save_model(self, model_path, filename):
        """
        NMF 모델을 저장하는 메소드

        :param model_path: NMF 모델의 저장 경로
        :param filename: NMF 모델의 파일 이름
        :return:
        """
        pkl.dump(self.NMF, open(model_path+"/"+filename, 'wb'), pkl.HIGHEST_PROTOCOL)

    def load_model(self, model_path, filename):
        """
        NMF 모델을 불러오는 메소드

        :param model_path: 불러올 NMF 모델의 경로
        :param filename: 불러올 NMF 모델 파일의 이름
        :return:
        """
        self.NMF = pkl.load(open(model_path + "/" + filename, 'rb'))

    def transform(self, head, body, with_cos=True):
        """
        학습된 NMF에 TFIDF 벡터를 넣어 NMF vector로 변환시키는 메소드

        :param head: 변환시킬 head TFIDF 벡터
        :param body: 변환시킬 body TFIDF 벡터
        :return:
            transformed_head : NMF vector로 변환 된 head
            transformed_body : NMF vector로 변환 된 body
            combined : 변환 된 head-body pair NMF vector
        """

        transformed_head = self.NMF.transform(head)
        transformed_body = self.NMF.transform(body)
        combined = np.concatenate((transformed_head.toarray(), transformed_body.toarray()), axis=1)

        return transformed_head, transformed_body, combined

    def transform_and_save_data(self, head, body, save_path, filename):
        """
        학습된 TFIDFVectorizer에 문장을 넣어 TFIDF vector로 변환 및 저장하는 메소드

        :param head: 변환시킬 head 문장들
        :param body: 변환시킬 body 문장들
        :param save_path: 저장할 경로
        :param filename: 저장할 파일 이름
        :return:
        """

        transformed_head = self.NMF.transform(head).toarray()
        transformed_body = self.NMF.transform(body).toarray()
        combined = np.concatenate((transformed_head.toarray(), transformed_body.toarray()), axis=1)

        head_file_name = save_path+"/"+filename+"_head.pkl"
        body_file_name = save_path+"/"+filename+"_body.pkl"
        combined_file_name = save_path+"/"+filename+"_combined.pkl"

        print('Saving transformed TFIDF head body vectors...')
        pkl.dump(transformed_head, open(head_file_name, 'wb'), pkl.HIGHEST_PROTOCOL)
        pkl.dump(transformed_body, open(body_file_name, 'wb'), pkl.HIGHEST_PROTOCOL)
        pkl.dump(combined, open(combined_file_name, 'wb'), pkl.HIGHEST_PROTOCOL)
        print('Saving finish TFIDF head body vectors')
        print('Head : {}\n Body : {}\n Combined : {}\n'
              .format(head_file_name, body_file_name, combined_file_name))

# if __name__ == "__main__":
    # count_vectorizer = CountVector_generator\
    #     (max_features=5000, analyzer='word', ngram_range=(1, 1), stop_words='english')

