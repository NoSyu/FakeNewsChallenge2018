from Tree_models.utils.get_input_datas import get_head_body_tuples, get_head_body_tuples_test, get_y_labels
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from Tree_models.utils.score import report_score
head_train, body_train = get_head_body_tuples()
head_test, body_test = get_head_body_tuples_test()

train_y, test_y = get_y_labels()

count_vec = CountVectorizer(analyzer='word', ngram_range=(1, 1), stop_words='english',
                            max_features=2500)
count_vec.fit([h+". "+b for h, b in zip(head_train, body_train)])

# count_vocab = count_vec.vocabulary_
print('count_vec ...')
head_train = count_vec.transform(head_train)
body_train = count_vec.transform(body_train)
head_test = count_vec.transform(head_test)
body_test = count_vec.transform(body_test)
print('count_vec finish...')

# print(head_train)
train_data = np.concatenate((head_train.toarray(), body_train.toarray()), axis=1)
test_data = np.concatenate((head_test.toarray(), body_test.toarray()), axis=1)
# print('train Decision tree')
clf = DecisionTreeClassifier()
clf.fit(train_data, train_y)
# print('train Naive bayes')
# clf=MultinomialNB()
# clf.fit(train_data, np.argmax(train_y, axis=1))
predicted = clf.predict(test_data)
predicted = np.argmax(predicted, axis=1)
test_y = np.argmax(test_y, axis=1)

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
LABELS_RELATED = ['unrelated','related']
RELATED = LABELS[0:3]
report_score([LABELS[e] for e in test_y], [LABELS[e] for e in predicted])