import pickle as pkl
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy import sparse
from scipy.sparse import hstack

def cos_sim_generator(filepath, head_filename, body_filename, save_filepath, save_filename):
    """
    head와 body의 cosine similarity를 구해 combine 하는 함수

    :param filepath: 읽을 파일의 경로
    :param head_filename: head pkl 파일의 경로
    :param body_filename: body pkl 파일의 경로
    :param save_filepath: 저장할 파일의 경로
    :param save_filename: 저장할 파일의 이름
    :return: 
    """
    def load_pkl(file_path, filename):
        print('Loading {}/{}'.format(file_path, filename))
        pkl_file = pkl.load(open(file_path+"/"+filename, 'rb'))
        print('Load {}/{} finish!'.format(file_path, filename))
        return pkl_file

    def save_pkl(save_file, save_filepath, save_filename):
        pkl.dump(save_file, open(save_filepath + "/" + save_filename, 'wb'), pkl.HIGHEST_PROTOCOL)

    head = load_pkl(filepath, head_filename).toarray()
    body = load_pkl(filepath, body_filename).toarray()

    cos = []
    count = 0
    for h, b in zip(head, body):
        if count % 5000 == 0 and not count == 0:
            print('cos calculate count {}'.format(count))
        cos.append(cosine_similarity(h.reshape(1, -1), b.reshape(1, -1))[0])
        count += 1
    cos = np.array(cos)
    combined_cos = np.hstack([head, body, cos])
    combined_cos = sparse.csr_matrix(combined_cos)

    print('Saving cos sim combined file...')
    save_pkl(combined_cos, save_filepath, save_filename)
    print('Saving cos sim combined file finish!')

def nmf_cos_sim_generator(filepath, head_filename, body_filename, save_filepath, save_filename):
    """
    head와 body의 cosine similarity를 구해 combine 하는 함수

    :param filepath: 읽을 파일의 경로
    :param head_filename: head pkl 파일의 경로
    :param body_filename: body pkl 파일의 경로
    :param save_filepath: 저장할 파일의 경로
    :param save_filename: 저장할 파일의 이름
    :return:
    """
    def load_pkl(file_path, filename):
        print('Loading {}/{}'.format(file_path, filename))
        pkl_file = pkl.load(open(file_path+"/"+filename, 'rb'))
        print('Load {}/{} finish!'.format(file_path, filename))
        return pkl_file

    def save_pkl(save_file, save_filepath, save_filename):
        pkl.dump(save_file, open(save_filepath + "/" + save_filename, 'wb'), pkl.HIGHEST_PROTOCOL)

    head = load_pkl(filepath, head_filename).toarray()
    body = load_pkl(filepath, body_filename).toarray()

    cos = []
    count = 0
    for h, b in zip(head, body):
        if count % 5000 == 0 and not count == 0:
            print('cos calculate count {}'.format(count))
        cos.append(cosine_similarity(h.reshape(1, -1), b.reshape(1, -1))[0])
        count += 1
    cos = np.array(cos)
    combined_cos = np.hstack([head, body, cos])
    combined_cos = sparse.csr_matrix(combined_cos)

    print('Saving cos sim combined file...')
    save_pkl(combined_cos, save_filepath, save_filename)
    print('Saving cos sim combined file finish!')

def cos_sim_csr_generator(filepath, head_filename, body_filename, save_filepath, save_filename):
    """
    head와 body의 cosine similarity를 구해 combine 하는 함수

    :param filepath: 읽을 파일의 경로
    :param head_filename: head pkl 파일의 경로
    :param body_filename: body pkl 파일의 경로
    :param save_filepath: 저장할 파일의 경로
    :param save_filename: 저장할 파일의 이름
    :return: 
    """
    def load_pkl(file_path, filename):
        print('Loading {}/{}'.format(file_path, filename))
        pkl_file = pkl.load(open(file_path+"/"+filename, 'rb'))
        print('Load {}/{} finish!'.format(file_path, filename))
        return pkl_file

    def save_pkl(save_file, save_filepath, save_filename):
        pkl.dump(save_file, open(save_filepath + "/" + save_filename, 'wb'), pkl.HIGHEST_PROTOCOL)

    head = load_pkl(filepath, head_filename)
    body = load_pkl(filepath, body_filename)

    matrix_len = head.shape[0]

    cos = []
    for count, i in enumerate(range(matrix_len)):
        if count % 5000 == 0 and not count == 0:
            print('cos calculate count {}'.format(count))
        cos.append(cosine_similarity(head.getrow(i), body.getrow(i))[0])

    cos = sparse.csr_matrix(cos)
    combined_cos = hstack([head, body, cos], format='csr')

    print('Saving cos sim combined file...')
    save_pkl(combined_cos, save_filepath, save_filename)
    print('Saving cos sim combined file finish!')

def cos_generator(filepath, head_filename, body_filename, save_filepath, save_filename):
    cos_sim_generator(filepath=filepath, head_filename=head_filename, body_filename=body_filename,
                      save_filepath=save_filepath, save_filename=save_filename)


def load_pkl_sentiment(file_path, filename):
    print('Loading {}/{}'.format(file_path, filename))
    pkl_file = pkl._Unpickler(open(file_path+"/"+filename, 'rb'))
    pkl_file.encoding = 'latin1'
    pkl_file = pkl_file.load()
    print('Load {}/{} finish!'.format(file_path, filename))
    return pkl_file

if __name__ == "__main__":
    # print(np.array([1, 2, 3]).reshape(1, -1))
    # print(cosine_similarity([[1, 2, 3]], [[2, 3, 4]])[0])
    # a = np.array([[1], [2], [3]])
    # b = np.array([[4], [5], [6]])
    # c = np.array([10, 20, 30])

    # TF_IDF combine example
    file_path = '../../pickled_data'
    # senti_test = load_pkl(file_path, 'test.combined.senti.pkl')
    # print(senti_train)
    head_train = 'nmf_100_include_holdout_train_head.pkl'
    body_train = 'nmf_100_include_holdout_train_body.pkl'
    head_test = 'nmf_100_include_holdout_test_head.pkl'
    body_test = 'nmf_100_include_holdout_test_body.pkl'
    # head_train = 'svd_100_include_holdout_train_head.pkl'
    # body_train = 'svd_100_include_holdout_train_body.pkl'
    # head_test = 'svd_100_include_holdout_test_head.pkl'
    # body_test = 'svd_100_include_holdout_test_body.pkl'
    # head_train = 'tfidf_feat_1st_train_include_test_head.pkl'
    # body_train = 'tfidf_feat_1st_train_include_test_body.pkl'
    # head_test = 'tfidf_feat_1st_test_include_test_head.pkl'
    # body_test = 'tfidf_feat_1st_test_include_test_body.pkl'

    save_filepath = '../../saved_model'

    cos_sim_csr_generator\
        (file_path, head_train, body_train, save_filepath, 'nmf_100_cos_train_include_holdout.pkl')
    cos_sim_csr_generator\
        (file_path, head_test, body_test, save_filepath, 'nmf_100_cos_test_include_holdout.pkl')


    # head_name = 'tfidf_1st_5000_head.pkl'
    # body_name = 'tfidf_1st_5000_body.pkl'
    # save_file_name = 'tfidf_1st_5000_cosine.pkl'
    #
    # head_test_name = 'tfidf_1st_5000_test_head.pkl'
    # body_test_name = 'tfidf_1st_5000_test_body.pkl'
    # save_file_test_name = 'tfidf_1st_5000_test_cosine.pkl'
    # cos_generator(filepath=file_path, head_filename=head_name, body_filename=body_name,
    #                   save_filepath=file_path, save_filename=save_file_name)
    # cos_generator(filepath=file_path, head_filename=head_test_name, body_filename=body_test_name,
    #                     save_filepath=file_path, save_filename=save_file_test_name)

    # head_name = 'count_5000_head.pkl'
    # body_name = 'count_5000_body.pkl'
    # save_file_name = 'count_5000_cosine.pkl'
    #
    # head_test_name = 'count_5000_test_head.pkl'
    # body_test_name = 'count_5000_test_body.pkl'
    # save_file_test_name = 'count_5000_test_cosine.pkl'
    # tfidf_cos_generator(filepath=file_path, head_filename=head_name, body_filename=body_name,
    #                   save_filepath=file_path, save_filename=save_file_name)
    # tfidf_cos_generator(filepath=file_path, head_filename=head_test_name, body_filename=body_test_name,
    #                     save_filepath=file_path, save_filename=save_file_test_name)

    # head_name = 'nmf_200_head.pkl'
    # body_name = 'nmf_200_body.pkl'
    # save_file_name = 'nmf_200_cosine.pkl'
    #
    # head_test_name = 'nmf_200_test_head.pkl'
    # body_test_name = 'nmf_200_test_body.pkl'
    # save_file_test_name = 'nmf_200_test_cosine.pkl'
    # nmf_cos_sim_generator(filepath=file_path, head_filename=head_name, body_filename=body_name,
    #               save_filepath=file_path, save_filename=save_file_name)
    # nmf_cos_sim_generator(filepath=file_path, head_filename=head_test_name, body_filename=body_test_name,
    #               save_filepath=file_path, save_filename=save_file_test_name)


