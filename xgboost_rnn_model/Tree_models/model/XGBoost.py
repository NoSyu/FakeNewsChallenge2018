# from scipy.sparse import csr_matrix, hstack
import numpy as np
import xgboost as xgb
from Tree_models.utils.score import report_score

class XGBoost:

    def __init__(self, max_depth=6, colsample_bytree = 0.6, subsample = 1.0, eta = 0.1, objective = 'multi:softprob',
                 num_class = 4, nthread = 14, silent = 1, eval_metric = 'mlogloss', seed = 2017):
        self.num_class = num_class
        self.param = {
            'max_depth': max_depth,  # (defalut 6)
            'colsample_bytree': colsample_bytree,
            'subsample': subsample,  # 트리마다의 관측 데이터 샘플링 비율, 값이 낮아질수록 under-fitting(default=1)
            'eta': eta,
            'objective': objective,
            'num_class': num_class,
            'nthread': nthread,
            'silent': silent,
            'eval_metric': eval_metric,
            'seed': seed
        }
        self.bst = xgb.Booster()

    def train(self, train_data, test_data, num_round = 1000, verbose_eval=10, early_stopping_rounds=50):
        watch_list = [(train_data, 'train'), (test_data, 'test')]
        print('xgboost train start!')
        self.bst = xgb.train(self.param, train_data, num_round, watch_list,
                             verbose_eval=verbose_eval, early_stopping_rounds=early_stopping_rounds)
        print('train finish!')

    def load_model(self, model_path, model_name):
        path = model_path+"/"+model_name
        print('model loading...')
        self.bst.load_model(path)
        print('model load finish!')

    def save_model(self, model_path, model_name):
        path = model_path+"/"+model_name
        print('model saving...')
        self.bst.save_model(path)
        print('model save finish! {}'.format(path))

    def predict(self, actual, test_data, feature_size):
        LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
        LABELS_RELATED = ['unrelated', 'related']
        RELATED = LABELS[0:3]
        predicted = self.bst.predict(test_data).reshape(feature_size, self.num_class)
        report_score([LABELS[int(e)] for e in actual], [LABELS[int(e)] for e in np.argmax(predicted, axis=1)])
        return predicted