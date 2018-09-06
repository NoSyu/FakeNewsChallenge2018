import csv
import numpy as np
from tqdm import tqdm
from data_helper.data_helper import data_prepro

class Dataset():
    """
    Dataset을 읽어오는 Class

    """
    def __init__(self, body_path, stance_path):
        """
        Class 초기화 함수
        :param body_path : 읽을 body 파일의 경로
        :param stance_path : 읽을 stance 파일의 경로
        """
        self.body_path = body_path
        self.stance_path = stance_path
        self.stance2idx = {'agree' : 0, 'disagree' : 1, 'discuss' : 2, 'unrelated' : 3}


    def read_combine(self):
        """
        body와 stance 데이터를 읽어 통합시키는 함수
        :return:
        head : 전처리 된 head들의 list
        body : 전처리 된 body들의 list
        stance : one-hot vector로 표현 된 label들의 list
        """
    # add headline
        self.data = []
        body_tmp = {}

        # head에 있는 body id의 번호와 body의 본문을 matching해 dataset을 생성
        with open(self.body_path, encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=',')
            next(reader)    ## 첫줄 제외
            body_bar = tqdm(reader)
            for line in body_bar:
                body_bar.set_description("Processing {} pre-processing".format('body'))
                id, body = line
                id = int(id)
                body_tmp[id] = data_prepro(body)
                # body_tmp[id] = body

        with open(self.stance_path, encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=',')
            next(reader)    ## 첫줄 제외

            head_bar = tqdm(reader)
            for line in head_bar:
                head_bar.set_description("Processing {} pre-processing".format('head'))
                headline, id, stance = line
                headline = data_prepro(headline)
                id = int(id)
                self.data.append([headline, body_tmp[id], self.stance2idx[stance]])

        print('data size : ', len(self.data))
        print('read data finished!')
        self.data = np.array(self.data)
        head, body, stance = self.data[:, 0], self.data[:, 1], np.eye(4)[np.array(self.data[:, 2], dtype=np.int64)]
        return head, body, stance

    def read_tfidf_data(self):
        h, b = [], []

        with open(self.body_path, encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=',')
            next(reader)  ## 첫줄 제외
            body_bar = tqdm(reader)
            for line in body_bar:
                body_bar.set_description("Processing {} pre-processing".format('body'))
                id, body = line
                b.append(data_prepro(body))

        with open(self.stance_path, encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=',')
            next(reader)    ## 첫줄 제외

            head_bar = tqdm(reader)
            for line in head_bar:
                head_bar.set_description("Processing {} pre-processing".format('head'))
                headline, id, stance = line
                headline = data_prepro(headline)
                h.append(headline)

        return h, b

    def read_combine_test(self):
        """
        test용 함수
        :return:
        head : 전처리 된 head들의 list
        body : 전처리 된 body들의 list
        stance : one-hot vector로 표현 된 label들의 list
        """
    # add headline
        self.data = []
        body_tmp = {}
    
        count = 0
        # head에 있는 body id의 번호와 body의 본문을 matching해 dataset을 생성
        with open(self.body_path, encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=',')
            next(reader)    ## 첫줄 제외
            body_bar = tqdm(reader)
            for line in body_bar:
                body_bar.set_description("Processing {} pre-processing".format('body'))
                id, body = line
                id = int(id)
                # body_tmp[id] = data_prepro(body)
                body_tmp[id] = body

        with open(self.stance_path, encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=',')
            next(reader)    ## 첫줄 제외

            head_bar = tqdm(reader)
            for line in head_bar:
                head_bar.set_description("Processing {} pre-processing".format('head'))
                headline, id, stance = line
                headline = data_prepro(headline)
                id = int(id)
                self.data.append([headline, body_tmp[id], self.stance2idx[stance]])

                count += 1
                if count > 50:
                    break


        print('data size : ', len(self.data))
        print('read data finished!')
        self.data = np.array(self.data)
        head, body, stance = self.data[:, 0], self.data[:, 1], np.eye(4)[np.array(self.data[:, 2], dtype=np.int64)]
        return head, body, stance

if __name__ == "__main__":
    dataset = Dataset("../data/train_bodies.csv", "../data/train_stances.csv")
    head, body, stance = dataset.read_combine_test()