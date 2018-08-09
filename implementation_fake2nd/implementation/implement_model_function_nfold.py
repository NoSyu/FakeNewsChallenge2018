import tensorflow as tf
import numpy as np
from features.feature_generator import make_tfidf_combined_feature, load_tfidf_y
from utils.score import report_score

seed = 12345


def weight_variable(name, shape):
    return tf.get_variable(name=name, shape=shape,
                           initializer=tf.contrib.layers.variance_scaling_initializer
                           (factor=2.0, mode='FAN_IN', uniform=False, seed=seed))


def bias_variable(name, shape, bias):
    initial = tf.constant(bias, shape=shape)
    return tf.Variable(initial, name=name)


def MLP_Classifier(row_body, row_stance, head_dir, body_dir, label_dir, learning_rate=0.001, batch_size=188,
                   training_epoch=70,
                   init_bias=0.001, n_fold=10, mode='train', save_model_path='../tf_model/tfidf_5000_epoch'):
    """
    
    :param head_dir: head pkl 파일이 있는 경로
    :param body_dir: body pkl 파일이 있는 경로
    :param label_dir: y label pkl 파일이 있는 경로
    :param learning_rate: 학습률 파라미터
    :param batch_size: 배치 사이트 파라미터
    :param training_epoch: 학습 횟수인 epoch 파라미터
    :param init_bias: bias 초기값 파라미터
    :param n_fold: 몇 번의 validation test를 할 것인지에 대한 파라미터
    :param mode: train, test 두 모드를 선택하는 파라미터
    :param save_model_path: 모델이 저장될 경로를 입력하는 파라미터
    :return: 
    """
    lr = learning_rate
    batch_size = batch_size
    training_epoch = training_epoch
    hidden = (362, 942, 1071, 870, 318, 912, 247)

    n_classes = 4

    head_dir = head_dir
    body_dir = body_dir
    label_dir = label_dir
    init_bias = init_bias
    mode = mode

    X_data = make_tfidf_combined_feature(row_body, row_stance, head_dir, body_dir, label_dir)
    y_data = load_tfidf_y(label_dir)
    n_fold = n_fold

    LABELS = ['agree', 'disagree', 'discuss', 'unrelated']

    split_size = len(X_data) // n_fold
    predictions_list = []
    actual_list = []

    graph = tf.Graph()
    with graph.as_default():
        n_input = X_data.shape[1]
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

    for fold in range(n_fold):
        split_num = fold
        model_path = save_model_path + str(training_epoch) + '_n_fold_' + str(fold)
        # print(len(X_data))
        # print('sp num',split_num)
        # print('sp size', split_size)

        X_train = np.concatenate((X_data[:split_num * split_size], X_data[(split_num + 1) * split_size:]), axis=0)
        y_train = np.concatenate((y_data[:split_num * split_size], y_data[(split_num + 1) * split_size:]), axis=0)
        X_test = X_data[(split_num * split_size): (split_num + 1) * split_size]
        y_test = y_data[(split_num * split_size): (split_num + 1) * split_size]
        # print(len(X_train), len(X_test))

        print('X_train : ', X_train)
        print('X_test : ',X_test)
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
                        print(end)
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
                            print(fold, ' batch acc : ', batch_acc / batch_count)
                            print(fold, ' batch cost : ', batch_cost / batch_count)

                        batch_count += 1

                saver.save(sess, model_path)
                print('Training Finished!')

            if mode == 'test':
                saver.restore(sess, model_path)
                print('model load finish!')

                pred, acc = sess.run([predictions, accuracy], feed_dict={X: X_test, Y: y_test})
                print('fold : ', fold, ' pred :', pred, ', acc :', acc)
                report_score([LABELS[e] for e in np.argmax(y_test, 1)], [LABELS[e] for e in pred])
                actual_list += np.argmax(y_test, 1).tolist()
                predictions_list += pred.tolist()

    if mode == 'test':
        print('===========all datas scores===========')
        report_score([LABELS[e] for e in actual_list], [LABELS[e] for e in predictions_list])


if __name__ == "__main__":
    row_body = '../data/train_bodies.csv'
    row_stance = '../data/train_stances.csv'
    head_dir = '../pickled_model/tfidf_head_feature_train.pkl'
    body_dir = '../pickled_model/tfidf_body_feature_train.pkl'
    label_dir = '../pickled_model/tfidf_label_one_hot_train.pkl'
    save_model_path = '../tf_model/tfidf_5000_epoch'
    MLP_Classifier(row_body, row_stance, head_dir, body_dir, label_dir, learning_rate=0.001, batch_size=188,
                   training_epoch=70,
                   init_bias=0.001, n_fold=10, mode='train', save_model_path='../tf_model/tfidf_5000_epoch')
