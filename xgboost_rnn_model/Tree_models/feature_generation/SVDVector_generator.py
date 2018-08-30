from sklearn.decomposition import TruncatedSVD
import pickle as pkl
import numpy as np
from scipy.sparse import csr_matrix, vstack

class SVD_generator:

    def __init__(self, n_components=50, n_iter=15):
        """
        SVD 모델을 초기화하는 생성자

        """
        self.SVD = TruncatedSVD(n_components=n_components, n_iter=n_iter)

    def fit(self, combined_data):
        """
        SVD를 수행하는 메소드

        :param combined_data: 학습시킬 head body tfidf 벡터들
        :return:
        """
        print('fit Truncated SVD model...')
        self.SVD.fit(combined_data)
        print('fit Truncated SVD model finish!')

    def save_model(self, model_path, filename):
        """
        SVD 모델을 저장하는 메소드

        :param model_path: SVD 모델의 저장 경로
        :param filename: SVD 모델의 파일 이름
        :return:
        """
        pkl.dump(self.SVD, open(model_path+"/"+filename, 'wb'), pkl.HIGHEST_PROTOCOL)

    def load_model(self, model_path, filename):
        """
        NMF 모델을 불러오는 메소드

        :param model_path: 불러올 NMF 모델의 경로
        :param filename: 불러올 NMF 모델 파일의 이름
        :return:
        """
        self.SVD = pkl.load(open(model_path + "/" + filename, 'rb'))

    def transform(self, head, body):
        """
        학습된 SVD에 TFIDF 벡터를 넣어 NMF vector로 변환시키는 메소드

        :param head: 변환시킬 head TFIDF 벡터
        :param body: 변환시킬 body TFIDF 벡터
        :return:
            transformed_head : SVD vector로 변환 된 head
            transformed_body : SVD vector로 변환 된 body
            combined : 변환 된 head-body pair SVD vector
        """

        transformed_head = self.SVD.transform(head)
        transformed_body = self.SVD.transform(body)
        combined = vstack((transformed_head, transformed_body), format='csr')

        return transformed_head, transformed_body, combined

    def transform_and_save_data(self, head, body, save_path, filename, csr=True):
        """
        학습된 TFIDFVectorizer에 문장을 넣어 TFIDF vector로 변환 및 저장하는 메소드

        :param head: 변환시킬 head 문장들
        :param body: 변환시킬 body 문장들
        :param save_path: 저장할 경로
        :param filename: 저장할 파일 이름
        :param csr: csr 파일로 저장할 것인지 여부
        :return:
        """

        head_data = self.SVD.transform(head)
        body_data = self.SVD.transform(body)
        # transformed_data = self.NMF.transform(combined)
        # combined = vstack((transformed_head, transformed_body), format='csr')

        head_file_name = save_path+"/"+filename+"_head.pkl"
        body_file_name = save_path + "/" + filename + "_body.pkl"
        # combined_file_name = save_path+"/"+filename+"_combined.pkl"


        if csr:
            head_data = csr_matrix(head_data)
            body_data = csr_matrix(body_data)

        print('Saving transformed SVD head body vectors...')
        pkl.dump(head_data, open(head_file_name, 'wb'), pkl.HIGHEST_PROTOCOL)
        pkl.dump(body_data, open(body_file_name, 'wb'), pkl.HIGHEST_PROTOCOL)
        # pkl.dump(combined, open(combined_file_name, 'wb'), pkl.HIGHEST_PROTOCOL)
        print('Saving finish SVD head body vectors')
        print('Head : {}\nBody : {}\n'.format(head_file_name, body_file_name))


if __name__ == "__main__":
    # count_vectorizer = CountVector_generator\
    #     (max_features=5000, analyzer='word', ngram_range=(1, 1), stop_words='english')
    # from Tree_models.utils.get_input_datas import get_head_body_tuples, get_head_body_tuples_test
    def load_pkl(model_path, filename):
        print('Loading vector... {}'.format(filename))
        load_data = pkl.load(open(model_path + "/" + filename, 'rb'))
        print('Loading vector finish! {}'.format(filename))
        return load_data

    def execute_SVD_gen(max_features, model_path, head_train, body_train, head_test, body_test, save_model_path, save_model_name):
        h = load_pkl(model_path=model_path, filename=head_train)
        b = load_pkl(model_path=model_path, filename=body_train)

        h_test = load_pkl(model_path=model_path, filename=head_test)
        b_test = load_pkl(model_path=model_path, filename=body_test)

        filename = 'svd_'+str(max_features)+'_include_holdout'

        svd = SVD_generator(n_components=max_features)
        # svd.load_model(save_model_path, save_model_name)
        svd.fit(vstack((h, h_test, b, b_test), format='csr'))
        svd.save_model(save_model_path, save_model_name)

        svd.transform_and_save_data(h, b,
                                    save_path=model_path,
                                    filename = filename + '_train')
        svd.transform_and_save_data(h_test, b_test,
                                    save_path=model_path,
                                    filename = filename + '_test')

    max_features = 100
    model_path = '../../pickled_data'
    head_train = 'tfidf_feat_1st_train_include_test_head.pkl'
    body_train = 'tfidf_feat_1st_train_include_test_body.pkl'
    head_test = 'tfidf_feat_1st_test_include_test_head.pkl'
    body_test = 'tfidf_feat_1st_test_include_test_body.pkl'
    save_model_path = '../../saved_model'
    save_model_name = 'svd_50_1st_model_include_holdout.pkl'
    execute_SVD_gen(max_features = max_features, model_path=model_path,
                    head_train = head_train, body_train = body_train,
                    head_test = head_test, body_test = body_test,
                    save_model_path = save_model_path, save_model_name = save_model_name)
