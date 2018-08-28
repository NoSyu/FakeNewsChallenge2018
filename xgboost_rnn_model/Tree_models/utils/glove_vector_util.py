import numpy as np
from tqdm import tqdm
import pickle as pkl
from Tree_models.utils.get_input_datas import get_head_body_tuples, get_head_body_tuples_test

def load_Glove(glove_path, glove_file):
    print('Loading pre-trained Glove vector')

    with open(glove_path + '/' + glove_file) as f:
        model = {}
        lines = f.readlines()
        for line in tqdm(lines):
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
    print('Done')
    print(len(model)," words loaded!")
    return model

def sum_vectors(vectors):
    return np.array(np.sum(vectors, axis=0))

def save_summation_vectors(data_path, glove_path, glove_file, save_path, dim = 50):
    model = load_Glove(glove_path=glove_path, glove_file=glove_file)
    h_train, b_train = get_head_body_tuples(data_path=data_path)
    h_test, b_test = get_head_body_tuples_test(data_path=data_path)

    sum_head_train = []
    sum_body_train = []
    sum_head_test = []
    sum_body_test = []

    for h, b in tqdm(zip(h_train, b_train)):
        words_h = h.split()
        words_b = b.split()
        head_vectors = np.zeros(dim)
        body_vectors = np.zeros(dim)
        for wh, wb in zip(words_h, words_b):
            if wh in model.keys():
                head_vectors += model[wh]
            if wb in model.keys():
                body_vectors += model[wb]
            # print(body_vectors)


        sum_head_train.append(head_vectors)
        sum_body_train.append(body_vectors)
    # exit()
        # break
    for h, b in tqdm(zip(h_test, b_test)):
        words_h = h.split()
        words_b = b.split()
        head_vectors = np.zeros(dim)
        body_vectors = np.zeros(dim)
        for wh, wb in zip(words_h, words_b):
            if wh in model.keys():
                head_vectors += model[wh]
            if wb in model.keys():
                body_vectors += model[wb]

        sum_head_test.append(head_vectors)
        sum_body_test.append(body_vectors)

    sum_head_train, sum_body_train, sum_head_test, sum_body_test = \
        np.array(sum_head_train), np.array(sum_body_train), np.array(sum_head_test), np.array(sum_body_test)
    print(sum_head_train.shape)
    print(sum_body_train.shape)
    print(sum_head_test.shape)
    print(sum_body_test.shape)
    # np.hstack((sum_head_train, sum_body_train))
    # print(np.hstack((sum_head_train, sum_body_train)[0])
    #     # print(np.hstack((sum_head_train, sum_body_train)[1])))
    pkl.dump(np.hstack((sum_head_train, sum_body_train)),
             open(save_path+"/glove{}D_sum_head_body_train.pkl".format(dim), 'wb'), pkl.HIGHEST_PROTOCOL)
    print('file saved {}'.format(save_path+"/glove200D_sum_head_body_train.pkl"))

    pkl.dump(np.hstack((sum_head_test, sum_body_test)),
             open(save_path + "/glove{}D_sum_head_body_test.pkl".format(dim), 'wb'), pkl.HIGHEST_PROTOCOL)
    print('file saved {}'.format(save_path + "/glove200D_sum_head_body_test.pkl"))

if __name__ == "__main__":
    dim = 200
    data_path = '../../data'
    glove_path = '../../glove'
    save_path = '../../pickled_data'
    glove_file = 'glove.twitter.27B.{}d.txt'.format(dim)
    save_summation_vectors(data_path=data_path, glove_path=glove_path,
                           glove_file=glove_file, save_path=save_path, dim=dim)
    # model = load_Glove(glove_path=glove_path, glove_file=glove_file)
    # # print(model['train'])
    # # print(model['trained'])
    # head, body = get_head_body_tuples(data_path='../../data')
    # sum_head_vectors = []
    # sum_body_vectors = []
    # for h, b in tqdm(zip(head, body)):
    #     words_h = h.split()
    #     words_b = b.split()
    #     head_word_vectors = [model[w] for w in words_h if w in model.keys()]
    #     body_word_vectors = [model[w] for w in words_b if w in model.keys()]
    #     sum_head_vectors.append(sum_vectors(head_word_vectors))
    #     sum_body_vectors.append(sum_vectors(body_word_vectors))
    # print(np.array(sum_head_vectors).shape)
    # print(sum_head_vectors[0:2])
    # print(sum_body_vectors[0:2])
    # print(np.sum(np.array([[1, 2, 3, 4], [4, 5, 6, 7]]), axis=0))