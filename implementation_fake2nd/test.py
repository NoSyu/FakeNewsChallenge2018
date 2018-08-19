# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from utils.dataset import DataSet
# from utils.generate_test_splits import kfold_split, generate_hold_out_split, get_stances_for_folds
#
# x = np.array([[1, 2, 3, 4], [3, 2, 4, 1]])
# y = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
#
# # x = x.reshape(1, -1)
# # y = y.reshape(1, -1)
#
# print(x)
# print(y)
# for i, j in zip(x, y):
#     i = i.reshape(1, -1)
#     j = j.reshape(1, -1)
#     print(cosine_similarity(i, j))
# if __name__ == "__main__":
#     dataset = DataSet("./data")
#     # generate_hold_out_split(dataset)
#     folds, hold_out_ids = kfold_split(dataset, 0.8, 10)
#     #stance_folds : fold된 데이터 10등분 된 데이터 [ {fold0}, {fold1}, {fold2} ,,,, {fold9}]
#     stance_folds, stance_hold_out = get_stances_for_folds(dataset, folds, hold_out_ids)
#     print(len(stance_folds[0]))
#     print(len(stance_hold_out))
#     # print(stance_folds.items())
#     # print(stance_hold_out[:10])

a = [1, 2, 3, 4]
b = [4, 5 ,6, 7]
a.extend(b)
print(a)
