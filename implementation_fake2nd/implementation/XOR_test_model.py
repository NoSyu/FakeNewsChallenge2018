import tensorflow as tf
import numpy as np

seed = 12345
def weight_variable(name, shape):
    # return tf.get_variable(name=name, shape=shape,
    #                        initializer=tf.contrib.layers.variance_scaling_initializer
    #                        (factor=2.0, mode='FAN_IN', uniform=False, seed=seed))
    return tf.Variable(tf.random_normal(shape), name=name)

#
def bias_variable(name, shape):
    # initial = tf.constant(1e-3, shape=shape)
    # return tf.Variable(initial, name=name)
    return tf.Variable(tf.random_normal(shape), name=name)

lr = 0.001
batch_size = 4
training_epoch = 10000
# hidden = (362, 942, 1071, 870, 318, 912, 247)
# hidden = (600, 600, 600, 600)
hidden = (10, 10, 10, 10)
n_classes = 2
export_dir = '../tf_model/'

head_dir = '../pickled_model/tfidf_head_feature_train_holdout.pkl'
body_dir = '../pickled_model/tfidf_body_feature_train_holdout.pkl'
label_dir = '../pickled_model/tfidf_label_one_hot_train_holdout.pkl'
init_bias = 0.001
# mode = 'train'
mode = 'test'
# model_path = '../tf_model/ensemble_tfidf_5000_epoch70_n_fold_'
model_path = '../tf_model/XOR_test_'

# X_data = make_tfidf_combined_feature(head_dir, body_dir)
# y_data = load_tfidf_y(label_dir)
n_fold = 1

X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

X_test = np.array([[1, 1], [0, 0], [0, 1], [1, 0]])
y_test = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
# X_test = np.array([[0, 1], [0, 1], [0, 1]])
# y_test = np.array([[0, 1], [0, 1], [0, 1]])
# exit()
# X_train, y_train = tf.train.shuffle_batch([X_train, y_train], batch_size, len_X, 10000, seed=12345)

graph = tf.Graph()
with graph.as_default():
    n_input = X_train.shape[1]
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

    layer4 = tf.nn.relu(tf.add(tf.matmul(layer3, weight_variable('w4', [hidden[2], hidden[3]])),
                               bias_variable('b4', [hidden[3]])))
    logits = tf.add(tf.matmul(layer4,weight_variable('out_w',[hidden[3], n_classes])),
                                  bias_variable('out_b',[n_classes]))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_tensor).minimize(cost)

    predictions = tf.argmax(logits, 1)
    correct_prediction = tf.equal(predictions, tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()

split_num = 1
split_size = len(X_train) // 10

for fold in range(n_fold):
    model_path = model_path+str(fold)

    # np.random.seed(seed)
    # np.random.shuffle(X_train)
    # np.random.seed(seed)
    # np.random.shuffle(y_train)
    #
    # print(X_train[0])
    # print(y_train[0])
    len_X = len(X_train)
    fold_count = len_X // 10
    tf.reset_default_graph()
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())


        if mode == 'train':
            print('Learning Started!')
            for epoch in range(training_epoch):

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
                calc_momentum = momentum_start
                                # + (float((momentum_end - momentum_start) / training_epoch) * epoch)
                #
                # if epoch > 0 and (epoch == 20 or epoch == 35 or epoch == 45):
                #     calc_learning_rate = float(calc_learning_rate / 10.0)

                # print(ep)
                c, acc, lo,  _ = sess.run([cost, accuracy, logits, optimizer], feed_dict={X: X_train, Y: y_train,
                                                                  learning_rate_tensor: lr, momentum:calc_momentum})
                    # print('cost : ', c, ' accuracy : ', acc)
                if epoch % 500 == 0:
                    print('cost : ', c)
                    print('logit : ', lo)



            saver.save(sess, model_path)
            print('Training Finished!')

        if mode =='test':
            scores = []
            for f in range(n_fold):
                # new_saver = tf.train.import_meta_graph(model_path+".meta")
                # saver.restore(sess, model_path)
                print('model load finish!')

                # print('Epoch', ep + 1, 'completed out of', ep, 'loss:', epoch_loss, 'LR=',calc_learning_rate)
                pred, acc, logits = sess.run([predictions, accuracy, logits], feed_dict={X: X_test, Y: y_test})
                # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_test, 1))
                print(pred, acc, logits)
                # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                # accuracy = sess.run(accuracy, feed_dict={X: X_test, Y: y_test})

            #     scores.append(accuracy)
            # print('n-fold accuracy: ', sum(scores)/n_fold)


