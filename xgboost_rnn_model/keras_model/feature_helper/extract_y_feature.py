import pandas as pd
import pickle as pkl
import numpy as np

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
def saved_y_data(path, filename, save_path, save_filename):
    y_data = []
    dp = pd.read_csv(path+"/"+filename, delimiter=',')
    for i in dp['Stance']:
        y_data.append(LABELS.index(i))
    y_data = np.array(y_data)
    pkl.dump(y_data, open(save_path+"/"+save_filename, 'wb'), pkl.HIGHEST_PROTOCOL)

def load_y_data(path, filename):
    return pkl.load(open(path+"/"+filename, 'rb'))

if __name__=='__main__':
    path = '../../data'
    save_path = '../../features'
    filename = 'train_stances.csv'
    s_filename = 'train_y_label.pkl'
    saved_y_data(path, filename, save_path, s_filename)