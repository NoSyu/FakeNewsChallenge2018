from Tree_model_try.utils.get_input_datas import get_y_labels
from Tree_model_try.tree_model.Decision_Tree import Decision_Tree
import pickle as pkl

def load_pkl(file_path, filename):
    print('Loading {}/{}'.format(file_path, filename))
    pkl_file = pkl.load(open(file_path + "/" + filename, 'rb'))
    print('Load {}/{} finish!'.format(file_path, filename))
    return pkl_file

model = 'tfidf_cos_5000'
file_path = '../pickled_data'
model_path = '../saved_model'

if model == 'tfidf_cos_5000':

    train_X_filename = 'tfidf_5000_cosine.pkl'
    train_y_filename = 'train_y_label.pkl'
    test_X_filename = 'tfidf_5000_test_cosine.pkl'
    test_y_filename = 'test_y_label.pkl'

    train_X = load_pkl(file_path, train_X_filename).toarray()
    test_X = load_pkl(file_path, test_X_filename).toarray()
    train_y, test_y = get_y_labels(file_path, one_hot=True)

    clf = Decision_Tree()
    clf.fit(train_X, train_y)
    clf.save_model(save_file_path=model_path,
                   model_name='decision_tree_5000_cosine.pkl')
    clf.predict_and_scoring(test_X, test_y)

elif model == 'count_5000':
    train_X_filename = 'count_5000_combined.pkl'
    train_y_filename = 'train_y_label.pkl'
    test_X_filename = 'count_5000_test_combined.pkl'
    test_y_filename = 'test_y_label.pkl'

    train_X = load_pkl(file_path, train_X_filename).toarray()
    test_X = load_pkl(file_path, test_X_filename).toarray()
    train_y, test_y = get_y_labels(file_path, one_hot=True)

    clf = Decision_Tree()
    clf.fit(train_X, train_y)
    clf.save_model(save_file_path=model_path,
                   model_name='decision_tree_count_5000.pkl')
    clf.predict_and_scoring(test_X, test_y)

elif model =='count_cos_5000':
    train_X_filename = 'count_5000_cosine.pkl'
    train_y_filename = 'train_y_label.pkl'
    test_X_filename = 'count_5000_test_cosine.pkl'
    test_y_filename = 'test_y_label.pkl'

    train_X = load_pkl(file_path, train_X_filename).toarray()
    test_X = load_pkl(file_path, test_X_filename).toarray()
    train_y, test_y = get_y_labels(file_path, one_hot=True)

    clf = Decision_Tree()
    clf.fit(train_X, train_y)
    clf.save_model(save_file_path=model_path,
                   model_name='decision_tree_count_cosine_5000.pkl')
    clf.predict_and_scoring(test_X, test_y)

elif model == 'naive_tfidf_5000':
    from sklearn.naive_bayes import MultinomialNB
    from Tree_model_try.utils.score import report_score
    import numpy as np
    train_X_filename = 'tfidf_5000_combined.pkl'
    train_y_filename = 'train_y_label.pkl'
    test_X_filename = 'tfidf_5000_test_combined.pkl'
    test_y_filename = 'test_y_label.pkl'

    train_X = load_pkl(file_path, train_X_filename).toarray()
    test_X = load_pkl(file_path, test_X_filename).toarray()
    train_y, test_y = get_y_labels(file_path, one_hot=False)
    # print(train_X[0])
    # print(np.array(train_y).reshape(1, -1))
    clf = MultinomialNB()
    clf.fit(train_X, train_y)
    predicted = clf.predict(test_X)
    LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
    LABELS_RELATED = ['unrelated', 'related']
    RELATED = LABELS[0:3]
    report_score([LABELS[e] for e in test_y], [LABELS[e] for e in predicted])