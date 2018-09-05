import pickle as pkl
from sklearn.preprocessing import normalize
from scipy.sparse import hstack, vstack, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, TruncatedSVD
import xgboost as xgb

class XGBoostVectorizer:

    def __init__(self, BaseEstimator=None, TransformerMixin=None):

        self.baseEstimator = BaseEstimator
        self.transformerMixin = TransformerMixin

        self.modelPath = '../vectorizer_model'
        self.countVectorizer = None
        self.tfidfVectorizer = None
        self.SVDVectorizer = None
        self.NMFVectorizer = None

    def fit(self, X):
        """
            Vectorizer 모델을 새로운 데이터로 train하는 메소드

        :param X: [[head sentence, body sentence], [head2, body2], ..., [headN, bodyN]] 형식의 2차원 리스트
        :return:
        """
        head = [x[0] for x in X]
        body = [x[1] for x in X]

        self.countVectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 3), stop_words='english')
        print('Fit CountVectorizer...')
        self.countVectorizer.fit([h + ". " + b for h, b in zip(head, body)])
        print('Done')

        self.tfidfVectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), stop_words='english')
        print('Fit TFIDFVectorizer...')
        self.tfidfVectorizer.fit([h + ". " + b for h, b in zip(head, body)])
        print('Done')
        print(self.tfidfVectorizer.vocabulary_)

        tfidf_h_vec = csr_matrix(self.tfidfVectorizer.transform(head))
        tfidf_b_vec = csr_matrix(self.tfidfVectorizer.transform(body))

        self.SVDVectorizer = TruncatedSVD(n_components=50, n_iter=15)
        print('Fit SVDVectorizer...')
        self.SVDVectorizer.fit(vstack((tfidf_h_vec, tfidf_b_vec), format='csr'))
        print('Done')

        # fit할 때 시간이 오래걸림
        self.NMFVectorizer = NMF(n_components=200, init='nndsvd', verbose=True)
        print('Fit NMFVectorizer...')
        self.NMFVectorizer.fit(vstack((tfidf_h_vec, tfidf_b_vec), format='csr'))
        print('Done')


    def transform(self, X):
        """
            pretrain된 모델이나 새롭게 fit한 feature engineering 모델을 이용해
            input을 vectorizing 하는 메소드

        :param X: [[head sentence, body sentence], [head2, body2], ..., [headN, bodyN]] 형식의 2차원 리스트
        :return: 모든 feature들을 concat한 csr matrix

        countVector head + body : (n, 892076)
        tfidfVector head + body + cos : (n, 892077)
        SVDVector 100 head + body + cos : (n, 201)
        NMFVector 200 head + body + cos : (n, 401)
        pre-trained model data input shape : (n, 3568907)

        새로 fit한 경우는 다른 shape가 입력
        """

        def cos_sim(head, body):
            head, body = csr_matrix(head), csr_matrix(body)
            cos_sim_data = []
            for i in range(head.shape[0]):
                cos_sim_data.append(cosine_similarity(head.getrow(i), body.getrow(i))[0])
            return csr_matrix(cos_sim_data)

        head = [x[0] for x in X]
        body = [x[1] for x in X]
        print(head)
        print(body)
        if self.countVectorizer is None:
            self.countVectorizer = pkl.load(open(self.modelPath + '/count_vectorizer_model.pkl', 'rb'))
            self.tfidfVectorizer = pkl.load(open(self.modelPath + '/tfidf_vectorizer_model.pkl', 'rb'))
            self.SVDVectorizer = pkl.load(open(self.modelPath + '/svd_100_vectorizer_model.pkl', 'rb')) #파일 교체해야함
            self.NMFVectorizer = pkl.load(open(self.modelPath + '/nmf_200_vectorizer_model.pkl', 'rb'))

        count_h_vec = self.countVectorizer.transform(head)
        count_b_vec = self.countVectorizer.transform(body)

        tfidf_h_vec = self.tfidfVectorizer.transform(head)
        tfidf_b_vec = self.tfidfVectorizer.transform(body)
        tfidf_cos_vec = cos_sim(tfidf_h_vec, tfidf_b_vec)

        svd_h_vec = self.SVDVectorizer.transform(tfidf_h_vec)
        svd_b_vec = self.SVDVectorizer.transform(tfidf_b_vec)
        svd_cos_vec = cos_sim(svd_h_vec, svd_b_vec)

        nmf_h_vec = self.NMFVectorizer.transform(tfidf_h_vec)
        nmf_b_vec = self.NMFVectorizer.transform(tfidf_b_vec)
        nmf_cos_vec = cos_sim(nmf_h_vec, nmf_b_vec)
        # print(count_h_feature.shape)
        # print(count_b_feature.shape)
        # print(tfidf_h_feature.shape)
        # print(tfidf_b_feature.shape)
        # print(svd_h_feature.shape)
        # print(svd_cos)

        features = normalize(hstack((count_h_vec, count_b_vec, tfidf_h_vec, tfidf_b_vec, tfidf_cos_vec,
                                 svd_h_vec, svd_b_vec, svd_cos_vec, nmf_h_vec, nmf_b_vec, nmf_cos_vec), format='csr'))
        xgb_feature = xgb.DMatrix(features)
        return xgb_feature

if __name__ == "__main__":
    # 이 데이터로 fit 메소드 test 시에는 nmf 값을 10으로 주고 시도했습니다.
    # 옳은 예시가 아님을 밝힙니다.

    test_X = [["hello", "hello my name is lee"],
              ["bye", "bye my name is lee."],
              ["car", "bus car train"],
              ["train", "apple, train, bus"],
              ["bow", "sword, news, bow"],
              ["subway", "news, boy, line"],
              ["pot", "cow, soccer"],
              ["row", "low, huge, big, large"],
              ["much", "bird"],
              ["poor", "quite"]]
    test_y = [1, 2, 1, 0, 1, 1, 1, 1, 0, 3]
    xgb_vector = XGBoostVectorizer()
    # xgb_vector.fit(test_X)
    print(xgb_vector.transform(test_X, test_y))

