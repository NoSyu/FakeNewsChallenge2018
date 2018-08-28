import pickle as pkl
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics.pairwise import cosine_similarity

def load_pkl_sentiment(file_path, filename):
    print('Loading {}/{}'.format(file_path, filename))
    pkl_file = pkl._Unpickler(open(file_path+"/"+filename, 'rb'))
    pkl_file.encoding = 'latin1'
    pkl_file = pkl_file.load()
    print('Load {}/{} finish!'.format(file_path, filename))
    return pkl_file

def save_pkl(save_file, save_filepath, save_filename):
    pkl.dump(save_file, open(save_filepath + "/" + save_filename, 'wb'), pkl.HIGHEST_PROTOCOL)

def calculate_cos_sim_and_save(h, b, save_path, save_filename):
    import numpy as np
    print(h.shape)
    print(b.shape)
    cos_list = []
    for head, body in zip(h, b):
        cos_list.append(cosine_similarity([head], [body])[0])

    h = csr_matrix(np.array(h))
    b = csr_matrix(np.array(b))
    cos_list = csr_matrix(np.array(cos_list))
    combined_data = hstack((h, b, cos_list), format='csr')

    save_pkl(combined_data, save_path, save_filename)


if __name__ == '__main__':

    file_path = '../../pickled_data'

    senti_train = load_pkl_sentiment(file_path, 'train.combined.senti.pkl')
    senti_test = load_pkl_sentiment(file_path, 'test.combined.senti.pkl')
    senti_h_train = senti_train[:, :4]
    senti_b_train = senti_train[:, 4:]
    senti_h_test = senti_test[:, :4]
    senti_b_test = senti_test[:, 4:8]

    # print(senti_train.shape)
    # print(senti_test.shape)
    # print(senti_h_test.shape)
    # print(senti_h_test[0])
    # print(senti_b_test[0])
    calculate_cos_sim_and_save(senti_h_train, senti_b_train, save_path=file_path, save_filename='senti_train_cos.pkl')
    calculate_cos_sim_and_save(senti_h_test, senti_b_test, save_path=file_path, save_filename='senti_test_cos.pkl')
    # print(senti_train.shape)