import tensorflow as tf
import numpy as np
from feature_helpers.feature_generator import make_tfidf_combined_feature_5000, load_tfidf_y
from utils.score import report_score

seed = 12345


def weight_variable(name, shape):
    return tf.get_variable(name=name, shape=shape,
                           initializer=tf.contrib.layers.variance_scaling_initializer
                           (factor=2.0, mode='FAN_IN', uniform=False, seed=seed))


def bias_variable(name, shape, bias):
    initial = tf.constant(bias, shape=shape)
    return tf.Variable(initial, name=name)

def shuffle_dataset(seed, data):
    print('data shuffling...')
    np.random.seed(seed)
    np.random.shuffle(data)
    print('data shuffling finished...')
    return data

def MLP_Classifier(row_body_train, row_stance_train, row_body_test, row_stance_test,
                   head_dir_train, body_dir_train, label_dir_train, head_dir_test, body_dir_test, label_dir_test,
                   learning_rate=0.001, batch_size=188,
                   training_epoch=70,
                   init_bias=0.001, mode='train', save_model_path='../tf_model/tfidf_5000_epoch', holdout=False):
    """

    :param row_head_train: 원본 head train 파일이 있는 경로
    :param row_stance_train: 원본 stance train 파일이 있는 경로
    :param row_head_test: 원본 head test  파일이 있는 경로
    :param row_stance_test: 원본 stance test 파일이 있는 경로
    :param head_dir_train: head train pkl 파일이 있는 경로
    :param body_dir_train: body train pkl 파일이 있는 경로
    :param label_dir_train: y train label pkl 파일이 있는 경로
    :param head_dir_test: head test pkl 파일이 있는 경로
    :param body_dir_test: body test pkl 파일이 있는 경로
    :param label_dir_test: y test label pkl 파일이 있는 경로
    :param learning_rate: 학습률 파라미터
    :param batch_size: 배치 사이트 파라미터
    :param training_epoch: 학습 횟수인 epoch 파라미터
    :param init_bias: bias 초기값 파라미터
    :param mode: train, test 두 모드를 선택하는 파라미터
    :param save_model_path: 모델이 저장될 경로를 입력하는 파라미터
    :return: 
    """
    lr = learning_rate
    batch_size = batch_size
    training_epoch = training_epoch
    hidden = (362, 942, 1071, 870, 318, 912, 247)

    n_classes = 4
    if mode == 'train':
        X_train = make_tfidf_combined_feature_5000(row_body_train, row_stance_train, head_dir_train, body_dir_train,
                                                   label_dir_train)
        y_train = load_tfidf_y(label_dir_train)
        # X_train = shuffle_dataset(seed, X_train)
        # y_train = shuffle_dataset(seed, y_train)
        if holdout:
            X_test = make_tfidf_combined_feature_5000(row_body_test, row_stance_test, head_dir_test, body_dir_test,
                                                      label_dir_test)
            y_test = load_tfidf_y(label_dir_test)
            # X_test = shuffle_dataset(seed, X_test)
            # y_test = shuffle_dataset(seed, y_test)
        n_input = X_train.shape[1]



    else:
        X_test = make_tfidf_combined_feature_5000(row_body_test, row_stance_test, head_dir_test, body_dir_test, label_dir_test)
        y_test = load_tfidf_y(label_dir_test)
        n_input = X_test.shape[1]

    LABELS = ['agree', 'disagree', 'discuss', 'unrelated']

    predictions_list = []
    actual_list = []

    graph = tf.Graph()
    with graph.as_default():

        X = tf.placeholder("float32", [None, n_input])
        Y = tf.placeholder("float32", [None, n_classes])
        learning_rate_tensor = tf.placeholder(tf.float32)
        momentum = tf.placeholder(tf.float32)

        layer1 = tf.nn.relu(tf.add(tf.matmul(X, weight_variable('w1', [n_input, hidden[0]])),
                                   bias_variable('b1', [hidden[0]], init_bias)))
        layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, weight_variable('w2', [hidden[0], hidden[1]])),
                                   bias_variable('b2', [hidden[1]], init_bias)))

        layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, weight_variable('w3', [hidden[1], hidden[2]])),
                                   bias_variable('b3', [hidden[2]], init_bias)))

        layer4 = tf.nn.relu(tf.add(tf.matmul(layer3,
                                             weight_variable('w4', [hidden[2], hidden[3]])),
                                   bias_variable('b4', [hidden[3]], init_bias)))

        layer5 = tf.nn.relu(tf.add(tf.matmul(layer4,
                                             weight_variable('w5', [hidden[3], hidden[4]])),
                                   bias_variable('b5', [hidden[4]], init_bias)))

        layer6 = tf.nn.relu(tf.add(tf.matmul(layer5,
                                             weight_variable('w6', [hidden[4], hidden[5]])),
                                   bias_variable('b6', [hidden[5]], init_bias)))

        layer7 = tf.nn.relu(tf.add(tf.matmul(layer6,
                                             weight_variable('w7', [hidden[5], hidden[6]])),
                                   bias_variable('b7', [hidden[6]], init_bias)))
        logits = tf.add(tf.matmul(layer7, weight_variable('out_w', [hidden[6], n_classes])),
                        bias_variable('out_b', [n_classes], init_bias))
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_tensor).minimize(cost)

        predictions = tf.argmax(logits, 1)
        correct_prediction = tf.equal(predictions, tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver(max_to_keep=10)

    model_path = save_model_path + str(training_epoch)

    tf.reset_default_graph()
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())

        if mode == 'train':
            print('Learning Started!')
            calc_learning_rate = lr
            for epoch in range(training_epoch):
                print('epoch : ', epoch)
                print(len(X_train))
                momentum_start = 0.5
                momentum_end = 0.99
                i = 0
                calc_momentum = momentum_start + (float((momentum_end - momentum_start) / training_epoch) * epoch)

                if epoch > 0 and (epoch == 20 or epoch == 35 or epoch == 45):
                    calc_learning_rate = float(calc_learning_rate / 10.0)

                batch_acc = 0.0
                batch_cost = 0.0
                batch_count = 1
                # print(ep)
                while i < len(X_train):

                    start = i
                    end = i + batch_size
                    batch_x = np.array(X_train[start:end])
                    batch_y = np.array(y_train[start:end])

                    c, acc, _ = sess.run([cost, accuracy, optimizer], feed_dict={X: batch_x, Y: batch_y,
                                                                                 learning_rate_tensor: calc_learning_rate,
                                                                                 momentum: calc_momentum})
                    # print('cost : ', c, ' accuracy : ', acc)
                    i += batch_size
                    batch_acc += acc
                    batch_cost += c
                    if i % (batch_size * 90) == 0:
                        print(i, '/', len(X_train))
                        print(' batch acc : ', batch_acc / batch_count)
                        print(' batch cost : ', batch_cost / batch_count)

                    batch_count += 1

                if holdout:
                    i = 0
                    while i < len(X_test):
                        start = i
                        end = i + batch_size
                        batch_x = np.array(X_test[start:end])
                        batch_y = np.array(y_test[start:end])

                        c, acc, _ = sess.run([cost, accuracy, optimizer], feed_dict={X: batch_x, Y: batch_y,
                                                                                     learning_rate_tensor: calc_learning_rate,
                                                                                     momentum: calc_momentum})
                        # print('cost : ', c, ' accuracy : ', acc)
                        i += batch_size
                        batch_acc += acc
                        batch_cost += c
                        if i % (batch_size * 90) == 0:
                            print(i, '/', len(X_test))
                            print(' batch acc : ', batch_acc / batch_count)
                            print(' batch cost : ', batch_cost / batch_count)
                        batch_count += 1

            saver.save(sess, model_path)
            print('Training Finished!')

        if mode == 'test':
            saver.restore(sess, model_path)
            print('model load finish!')

            pred, acc = sess.run([predictions, accuracy], feed_dict={X: X_test, Y: y_test})
            print('pred :', pred, ', acc :', acc)
            report_score([LABELS[e] for e in np.argmax(y_test, 1)], [LABELS[e] for e in pred])


if __name__ == "__main__":
    row_body_train = '../data/train_bodies.csv'
    row_stance_train = '../data/train_stances.csv'
    head_dir_train = '../pickled_model/tfidf_head_feature_train.pkl'
    body_dir_train = '../pickled_model/tfidf_body_feature_train.pkl'
    label_dir_train = '../pickled_model/tfidf_label_one_hot_train.pkl'

    row_body_test = '../data/competition_test_bodies.csv'
    row_stance_test = '../data/competition_test_stances.csv'
    head_dir_test = '../pickled_model/tfidf_head_feature_test.pkl'
    body_dir_test = '../pickled_model/tfidf_body_feature_test.pkl'
    label_dir_test = '../pickled_model/tfidf_label_one_hot_test.pkl'

    save_model_path = '../tf_model/tfidf_5000_epoch'
    MLP_Classifier(row_body_train, row_stance_train, row_body_test, row_stance_test,
                   head_dir_train, body_dir_train, label_dir_train,
                   head_dir_test, body_dir_test, label_dir_test,
                   learning_rate=0.001, batch_size=188,
                   training_epoch=70,
                   init_bias=0.001, mode='test', save_model_path=save_model_path, holdout=False)
