import tensorflow as tf
import numpy as np
from feature_helpers.feature_generator import make_tfidf_combined_feature_5000, load_tfidf_y
from utils.score import report_score
seed = 12345
def weight_variable(name, shape):
    return tf.get_variable(name=name, shape=shape,
                           initializer=tf.contrib.layers.variance_scaling_initializer
                           (factor=2.0, mode='FAN_IN', uniform=False, seed=seed))

#
def bias_variable(name, shape):
    initial = tf.constant(1e-3, shape=shape)
    return tf.Variable(initial, name=name)

lr = 0.001
batch_size = 188
training_epoch = 10
hidden = (362, 942, 1071, 870, 318, 912, 247)
# hidden = (600, 600, 600, 600)
# hidden = (10, 10, 10, 10)
n_classes = 4
export_dir = '../tf_model/'

head_dir = '../pickled_model/tfidf_head_feature_train_holdout.pkl'
body_dir = '../pickled_model/tfidf_body_feature_train_holdout.pkl'
label_dir = '../pickled_model/tfidf_label_one_hot_train_holdout.pkl'
init_bias = 0.001
# mode = 'train'
mode = 'test'
# model_path = '../tf_model/tfidf_5000_epoch'+str(training_epoch)+'_n_fold_'
# model_path = '../tf_model/XOR_test_'

X_data = make_tfidf_combined_feature_5000(head_dir, body_dir)
y_data = load_tfidf_y(label_dir)
n_fold = 10

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
LABELS_RELATED = ['unrelated', 'related']
RELATED = LABELS[0:3]

split_num = 0
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
    learning_rate_output = ""

    layer1 = tf.nn.relu(tf.add(tf.matmul(X, weight_variable('w1', [n_input, hidden[0]])),
                               bias_variable('b1', [hidden[0]])))
    layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, weight_variable('w2',[hidden[0], hidden[1]])),
                               bias_variable('b2',[hidden[1]])))

    layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, weight_variable('w3',[hidden[1], hidden[2]])),
                               bias_variable('b3',[hidden[2]])))

    layer4 = tf.nn.relu(tf.add(tf.matmul(layer3,
                                         weight_variable('w4', [hidden[2], hidden[3]])),
                               bias_variable('b4', [hidden[3]])))

    layer5 = tf.nn.relu(tf.add(tf.matmul(layer4,
                                         weight_variable('w5', [hidden[3], hidden[4]])),
                               bias_variable('b5', [hidden[4]])))

    layer6 = tf.nn.relu(tf.add(tf.matmul(layer5,
                                         weight_variable('w6', [hidden[4], hidden[5]])),
                               bias_variable('b6', [hidden[5]])))

    layer7 = tf.nn.relu(tf.add(tf.matmul(layer6,
                                         weight_variable('w7', [hidden[5], hidden[6]])),
                               bias_variable('b7', [hidden[6]])))
    logits = tf.add(tf.matmul(layer7,  weight_variable('out_w', [hidden[6], n_classes])),
                                       bias_variable('out_b', [n_classes]))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_tensor).minimize(cost)

    predictions = tf.argmax(logits, 1)
    correct_prediction = tf.equal(predictions, tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver(max_to_keep=10)


for fold in range(n_fold):
    split_num = fold
    model_path = '../tf_model/tfidf_5000_epoch'+str(training_epoch)+'_n_fold_'+str(fold)
    X_train = np.concatenate((X_data[:split_num*split_size], X_data[(split_num+1)*split_size:]), axis=0)
    y_train = np.concatenate((y_data[:split_num*split_size],y_data[(split_num+1)*split_size:]), axis=0)
    X_test = X_data[split_num*split_size : (split_num+1)*split_size]
    y_test = y_data[split_num*split_size : (split_num+1)*split_size]

    # np.random.seed(seed)
    # np.random.shuffle(X_train)
    # np.random.seed(seed)
    # np.random.shuffle(y_train)
    exit()
    print(X_train[0])
    print(y_train[0])
    len_X = len(X_train)
    fold_count = len_X // 10
    tf.reset_default_graph()
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())


        if mode == 'train':
            print('Learning Started!')
            for epoch in range(training_epoch):
                print('epoch : ', epoch)
                momentum_start = 0.5
                momentum_end = 0.99
                total_batch = len_X // batch_size
                # print('data shuffling...')
                # print('seed : ', epoch * 100 + 1)
                # np.random.seed(epoch)
                # np.random.shuffle(X_train)
                # np.random.seed(epoch)
                # np.random.shuffle(y_train)
                # print('data shuffling finished...')

                calc_learning_rate = lr
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
                                                                  learning_rate_tensor: lr, momentum:calc_momentum})
                    # print('cost : ', c, ' accuracy : ', acc)
                    i += batch_size
                    batch_acc += acc
                    batch_cost += c
                    if i % (batch_size*90) == 0:
                        print(i, '/', len(X_train))
                        print(fold, ' batch acc : ', batch_acc / batch_count)
                        print(fold, ' batch cost : ', batch_cost / batch_count)

                    batch_count += 1



            saver.save(sess, model_path)
            print('Training Finished!')

        if mode =='test':


            saver.restore(sess, model_path)
            scores = []
            print('model load finish!')

            # print('Epoch', ep + 1, 'completed out of', ep, 'loss:', epoch_loss, 'LR=',calc_learning_rate)
            pred, acc = sess.run([predictions, accuracy], feed_dict={X: X_test, Y: y_test})
            # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_test, 1))
            print('fold : ',fold,' pred :', pred,', acc :', acc)
            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # accuracy = sess.run(accuracy, feed_dict={X: X_test, Y: y_test})

            report_score([LABELS[e] for e in np.argmax(y_test, 1)], [LABELS[e] for e in pred])
            actual_list += np.argmax(y_test, 1).tolist()
            predictions_list += pred.tolist()


if mode == 'test':
    print('===========all datas scores===========')
    report_score([LABELS[e] for e in actual_list], [LABELS[e] for e in predictions_list])
            #     scores.append(accuracy)
            # print('n-fold accuracy: ', sum(scores)/n_fold)


