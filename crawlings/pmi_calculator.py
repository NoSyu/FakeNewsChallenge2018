import pandas as pd
import numpy as np
import pickle
import math
import string
import os
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

class PmiHelper():
    def __init__(self):
        self.total = 0
        self.instance = {'agree' : set(), 'disagree': set(), 'discuss': set(), 'unrelated': set()}
        self.wordset = set()
        self.word_dict = {'agree' : {}, 'disagree' : {}, 'discuss' : {}, 'unrelated' : {}}
        self.pmi_result = {'agree' : {}, 'disagree' : {}, 'discuss' : {}, 'unrelated' : {}}

    def read_csvfile(self, path, is_trian=True):
        print('csv file read start')
        savefile = './pickle_data/preprocessed_instance.pkl'
        if os.path.isfile(savefile):
            with open(savefile, 'rb') as outfile:
                self.instance = pickle.load(outfile)
        else:
            file = path + 'train' if is_trian else 'competition_test'
            headline = pd.read_csv(file + '_stances.csv')
            body = pd.read_csv(file + '_bodies.csv')
            data = pd.merge(headline, body, on='Body ID')
            for idx, row in data.iterrows():
                doc = row['Headline'] +' '+ row['articleBody']
                doc = doc.lower()
                doc = ''.join(character for character in doc if character not in string.punctuation)
                doc = ''.join(character for character in doc if character not in '0123456789')
                doc = ' '.join(doc.split())
                self.instance[row['Stance']].add(doc)
            with open(savefile, 'wb') as outfile:
                pickle.dump(self.instance, outfile, protocol=pickle.HIGHEST_PROTOCOL)

            print('make complete')

        for stance in self.instance:
            self.total += len(self.instance[stance])
        print('done\n')

        for stance in self.instance:
            print('{} : {}'.format(stance, len(self.instance[stance])))
        print('total size : {}\n'.format(self.total))


    def process(self, max_features=None):
        print('word dict process start')
        savefile = './pickle_data/dict_processedData('+'ALL' if max_features is None else max_features+').pkl'
        if os.path.isfile(savefile):
            with open(savefile, 'rb') as outfile:
                self.word_dict = pickle.load(outfile)
                self.wordset = pickle.load(outfile)
        else:
            for stance in self.instance:
                self.instance[stance] = list(self.instance[stance])
                couter_vector = CountVectorizer(max_features=max_features).fit(self.instance[stance])
                vector_array = couter_vector.transform(self.instance[stance]).toarray()
                vector_array[vector_array > 0] = 1
                vector_array = np.sum(vector_array, axis=0)
                for idx, word in enumerate(couter_vector.get_feature_names()):
                    self.wordset.add(word)
                    self.word_dict[stance][word] = vector_array[idx]
            with open(savefile, 'wb') as outfile:
                pickle.dump(self.word_dict, outfile, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.wordset, outfile, pickle.HIGHEST_PROTOCOL)

            print('make complete')

        print('word dict complete\n')

        print('word set size : {}'.format(len(self.wordset)))
        for stance in self.instance:
            print('{} dict : {}'.format(stance, len(self.word_dict[stance])))

    def calculate(self):
        for word in self.wordset:
            for stance in self.pmi_result:
                if word not in self.word_dict[stance]:
                    continue
                N00 = N01 = N10 = N11 = 1
                N11 = self.word_dict[stance][word]
                for stance_t in self.word_dict:
                    if word in self.word_dict[stance_t] and stance != stance_t:
                        N10 += self.word_dict[stance_t][word]
                    N00 += len(self.instance[stance_t])
                N01 = len(self.instance[stance]) - N11
                N00 = N00 - len(self.instance[stance]) - N10

                N = N00 + N01 + N10 + N11

                N_leftupper = (N11 / N) * math.log2((N * N11) / ((N10 + N11) * (N01 + N11)))
                N_rightupper = (N01 / N) * math.log2((N * N01) / ((N00 + N01) * (N01 + N11)))

                N_leftunder = (N10 / N) * math.log2((N * N10) / ((N10 + N11) * (N00 + N10)))
                N_rightunder = (N00 / N) * math.log2((N * N00) / ((N00 + N01) * (N10 + N00)))

                result = N_leftupper + N_rightupper + N_leftunder + N_rightunder
                self.pmi_result[stance][word] = result

    def show_graph(self):
        print("\n------------PMI RESULT-----------\n")
        for stance in self.pmi_result:
            sorted_data = sorted(self.pmi_result[stance].items(), key=lambda x:x[1], reverse=True)[:10]

            print(sorted_data)

            plt.title(stance.capitalize())
            plt.xlabel('Top 10 Words')
            plt.ylabel('FMI Score')
            plt.grid(True, axis='None')

            Sorted_Dict_Values = [x[1] for x in sorted_data]
            Sorted_Dict_Keys = [x[0] for x in sorted_data]

            plt.bar(range(len(sorted_data)), Sorted_Dict_Values, align='center')
            plt.xticks(range(len(sorted_data)), list(Sorted_Dict_Keys), rotation='70')

            plt.show()


if __name__ == '__main__':
    path = './data/'
    pmi = PmiHelper()
    pmi.read_csvfile(path)
    pmi.process(max_features=None)
    pmi.calculate()
    pmi.show_graph()