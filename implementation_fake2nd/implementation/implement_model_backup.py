import tensorflow as tf
import numpy as np
from feature_helpers.feature_generator import make_tfidf_combined_feature_5000, load_tfidf_y

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
training_epoch = 1
# hidden = (362, 942, 1071, 870, 318, 912, 247)
hidden = (600, 600, 600, 600)
n_classes = 4
export_dir = '../tf_model/'

head_dir = '../pickled_model/tfidf_head_feature_train_holdout.pkl'
body_dir = '../pickled_model/tfidf_body_feature_train_holdout.pkl'
label_dir = '../pickled_model/tfidf_label_one_hot_train_holdout.pkl'
init_bias = 0.001
mode = 'train'
# mode = 'test'
model_path = '../tf_model/ensemble_tfidf_5000_epoch70_n_fold_'

X_data = make_tfidf_combined_feature_5000(head_dir, body_dir)
y_data = load_tfidf_y(label_dir)
n_fold = 2
# exit()
# X_train, y_train = tf.train.shuffle_batch([X_train, y_train], batch_size, len_X, 10000, seed=12345)

n_input = X_data.shape[1]

class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_model()

    def _build_model(self):

        with tf.variable_scope(self.name):
            self.X = tf.placeholder("float32", [None, n_input])
            self.Y = tf.placeholder("float32", [None, n_classes])
            self.learning_rate_tensor = tf.placeholder(tf.float32)
            self.momentum = tf.placeholder(tf.float32)
            self.learning_rate_output = ""

            layer1 = tf.nn.relu(tf.add(tf.matmul(self.X,
                                                 weight_variable(self.name+'w1', [n_input, hidden[0]])),
                                       bias_variable(self.name+'b1', [hidden[0]])))

            layer2 = tf.nn.relu(tf.add(tf.matmul(layer1,
                                                 weight_variable(self.name+'w2',[hidden[0], hidden[1]])),
                                       bias_variable(self.name+'b2',[hidden[1]])))

            layer3 = tf.nn.relu(tf.add(tf.matmul(layer2,
                                                 weight_variable(self.name+'w3',[hidden[1], hidden[2]])),
                                       bias_variable(self.name+'b3',[hidden[2]])))

            layer4 = tf.nn.relu(tf.add(tf.matmul(layer3,
                                                 weight_variable(self.name + 'w4', [hidden[2], hidden[3]])),
                                       bias_variable(self.name + 'b4', [hidden[3]])))
            # layer4 = tf.nn.relu(tf.add(tf.matmul(layer3,
            #                                      weight_variable(self.name+'w4',[hidden[2], hidden[3]])),
            #                            bias_variable(self.name+'b4',[hidden[3]])))
            #
            # layer5 = tf.nn.relu(tf.add(tf.matmul(layer4,
            #                                      weight_variable(self.name+'w5',[hidden[3], hidden[4]])),
            #                            bias_variable(self.name+'b5',[hidden[4]])))
            #
            # layer6 = tf.nn.relu(tf.add(tf.matmul(layer5,
            #                                      weight_variable(self.name+'w6',[hidden[4], hidden[5]])),
            #                            bias_variable(self.name+'b6',[hidden[5]])))
            #
            # layer7 = tf.nn.relu(tf.add(tf.matmul(layer6,
            #                                      weight_variable(self.name+'w7',[hidden[5], hidden[6]])),
            #                            bias_variable(self.name+'b7',[hidden[6]])))
            # self.logits = tf.nn.softmax(tf.add(tf.matmul(layer7,
            #                                      weight_variable(self.name+'out_w',[hidden[6], n_classes])),
            #                                    bias_variable(self.name+'out_b',[n_classes])))

            self.logits = tf.nn.softmax(tf.add(tf.matmul(layer4,
                                                 weight_variable(self.name+'out_w',[hidden[3], n_classes])),
                                               bias_variable(self.name+'out_b',[n_classes])))
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_tensor).minimize(self.cost)

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.saver = tf.train.Saver()


    def predict(self, x_test):
        return self.sess.run(self.logits, feed_dict={self.X: x_test})

    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test})

    def train(self, x_data, y_data, learning_rate, momentum):
        return self.sess.run([self.cost, self.optimizer],
                             feed_dict={self.X: x_data, self.Y: y_data, self.learning_rate_tensor: learning_rate,
                                        self.momentum: momentum})
    def save(self, save_path):
        self.saver.save(sess, save_path)
        print('save model '+save_path)
    def load(self, model_path):
        self.saver.restore(self.sess, model_path)

split_num = 1
split_size = len(X_data) // 10

for fold in range(n_fold):
    model_path = '../tf_model/ensemble_tfidf_5000_epoch70_n_fold_'+str(fold)

    X_train = np.concatenate((X_data[:split_num*split_size], X_data[(split_num+1)*split_size:]), axis=0)
    y_train = np.concatenate((y_data[:split_num*split_size],y_data[(split_num+1)*split_size:]), axis=0)
    X_test = X_data[split_num*split_size : (split_num+1)*split_size]
    y_test = y_data[split_num*split_size : (split_num+1)*split_size]

    np.random.seed(seed)
    np.random.shuffle(X_train)
    np.random.seed(seed)
    np.random.shuffle(y_train)

    print(X_train[0])
    print(y_train[0])
    len_X = len(X_train)
    fold_count = len_X // 10

    with tf.Session(graph=tf.reset_default_graph()) as sess:

        if mode == 'train':
            models = []
            num_models = 5
            # for m in range(num_models):
            #     models.append(Model(sess, "model"+str(m)))

            sess.run(tf.global_variables_initializer())


            print('Learning Started!')
            for epoch in range(training_epoch):

                momentum_start = 0.5
                momentum_end = 0.99
                avg_cost_list = np.zeros(len(models))
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

                # print(ep)
                while i < len(X_train):
                    start = i
                    end = i + batch_size
                    batch_x = np.array(X_train[start:end])
                    batch_y = np.array(y_train[start:end])

                    for m_idx, m in enumerate(models):
                        c, _ = m.train(batch_x, batch_y, learning_rate=lr, momentum=calc_momentum)
                        avg_cost_list[m_idx] += c / total_batch

                        # print(epoch,', ', m_idx, 'accuracy : ', m.get_accuracy(X_test, y_test))
                    i += batch_size
                    if i % (batch_size*50) == 0:
                        print(i, '/', len(X_train))
                avg_accuracy = 0.0
                for ids, m in enumerate(models):
                    avg_accuracy += m.get_accuracy(X_test, y_test)
                    # m.save(model_path+"_epoch"+str(epoch)+"_model"+str(ids))
                print('avg accuracy : ', avg_accuracy/len(models))
                print('Epoch: ', epoch+1, ', cost = ', avg_cost_list)


            saver = tf.train.Saver()
            print('Training Finished!')
                    # _, c, out, y = sess.run([optimizer, cost, output, Y], feed_dict={
                    #     X: batch_x, Y: batch_y, momentum : calc_momentum, learning_rate_tensor: calc_learning_rate
                    # })
                    # if i % (batch_size*50) == 0 and i != 0:
                    #     print('epoch : {}, epoch_loss : {}, processing : {}/{}'.format(ep, c, i, len_X))
                    #     # print(out[:20])
                    #     # print(y[:20])
                    # epoch_loss += c

        if mode =='test':
            models = []
            num_models = 5
            # sess = saver.restore(sess, model_path+str(fold)+".ckpt")
            # for m in range(num_models):
            #     models.append(Model(sess, "model" + str(m)).load(model_path+"_epoch"+str(epoch)+"_model"+str(m)))

            print('model load finish!')

            # print('Epoch', ep + 1, 'completed out of', ep, 'loss:', epoch_loss, 'LR=',calc_learning_rate)
            predictions = np.zeros([fold_count, n_classes])
            print('fold', fold,'test')
            for m_idx, m in enumerate(models):
                print(m_idx, 'Accuracy: ', m.get_accuracy(X_test, y_test))
                p = m.predict(X_test)
                predictions += p

            ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_test, 1))
            ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
            print('Ensemble accuracy: ', sess.run(ensemble_accuracy))


