import tensorflow as tf
from util_LSTM.misc import get_sequence_data
from implementation.model_control import load_model
from feature_helpers.feature_generator import make_tfidf_combined_feature_cos_100_holdout
import numpy as np
from utils.score import report_score

def data_shuffle(data, seed):
    np.random.seed(seed)
    np.random.shuffle(data)
# data = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
# print(data, )

# exit()
LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
LABELS_RELATED = ['unrelated', 'related']
RELATED = LABELS[0:3]
training_epoch = 40
n_classes = 4
seq_len = 100
dim_size = 50
batch_size = 200
learning_rate = 0.001
hidden_size = 100

# base_data_path = '''os.path.dirname(os.path.dirname(__file__))''' + "/data"
# base_feat_path = '''os.path.dirname(os.path.dirname(__file__))''' + "/features"
# base_pickled_path = '''os.path.dirname(os.path.dirname(__file__))''' + "/pickled_model"


base_data_path = "../data"
base_feat_path = "../features"
base_pickled_path = "../pickled_model"

train_stance = base_data_path + "/train_stances.csv"
train_body = base_data_path + "/train_bodies.csv"
train_label = base_pickled_path + "/tfidf_label_one_hot_train.pkl"
train_head_dir = base_pickled_path + "/tfidf_100_head_holdout_train.pkl"
train_body_dir = base_pickled_path + "/tfidf_100_body_holdout_train.pkl"
train_cos_dir = base_pickled_path + "/cos_feat_100_holdout_train.pkl"

test_stance = base_data_path + "/competition_test_stances.csv"
test_body = base_data_path + "/competition_test_bodies.csv"
test_label = base_pickled_path + "/tfidf_label_one_hot_test.pkl"
test_head_dir = base_pickled_path + "/tfidf_cos_100_head_holdout_test.pkl"
test_body_dir = base_pickled_path + "/tfidf_cos_100_body_holdout_test.pkl"
test_cos_dir = base_pickled_path + "/cos_feat_100_holdout_test.pkl"

model_path = "../tf_lstm_model/LSTM_GloVe_50d_"+str(seq_len)+"_epoch"\
             +str(training_epoch)+"_batch"+str(batch_size)+"_holdout"

# mode = 'train'
mode = 'test'
# mode='fold_test'

embedding = np.load(base_feat_path+"/single_flat_LSTM_50d_100_embedding.npy")
dense_size = 600
dropout_keep_prob = 0.8
fold_size = 5

if mode == 'train':
    X_train_seq = get_sequence_data(train_body, train_stance, 'train')
    X_train_feat = make_tfidf_combined_feature_cos_100_holdout(train_body, train_stance, test_body, test_stance, train_head_dir, train_body_dir,
                                                       train_label, train_cos_dir)
    y_train = load_model(train_label)
    data_shuffle(X_train_seq, seed=12345)
    data_shuffle(X_train_feat, seed=12345)
    data_shuffle(y_train, seed=12345)
    input_len = len(X_train_feat[0])

elif mode == 'test':
    X_test_seq = get_sequence_data(test_body, test_stance, 'test')
    X_test_feat = make_tfidf_combined_feature_cos_100_holdout(test_body, test_stance, train_body, train_stance, test_head_dir, test_body_dir,
                                                      test_label, test_cos_dir)
    y_test = load_model(test_label)
    input_len = len(X_test_feat[0])

graph = tf.Graph()
with graph.as_default():
    X_seq = tf.placeholder(tf.int32, [None, seq_len])
    X_feat = tf.placeholder(tf.float32, [None, input_len])
    Y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    embedding_placeholder = tf.placeholder(tf.float32, [embedding.shape[0], embedding.shape[1]])

    def make_dropout_lstm_cell(hidden_size, keep_prob):
        lstm = tf.nn.rnn_cell.LSTMCell(hidden_size, initializer=tf.contrib.layers.xavier_initializer(), state_is_tuple=True)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
        return lstm_dropout

    lookup = tf.nn.embedding_lookup(embedding_placeholder, X_seq)
    # print(X_seq[:10])
    # print(embedding[:10])

    # print(X_seq)
    # print(Y)
    # print(keep_prob)
    # print(lookup)
    # lstm_dropout = tf.contrib.rnn.DropoutWrapper(make_dropout_lstm_cell(hidden_size, keep_prob), input_keep_prob=1-keep_prob, output_keep_prob=1-keep_prob)
    lstm_layers = tf.contrib.rnn.MultiRNNCell([make_dropout_lstm_cell(hidden_size, keep_prob) for _ in range(2)], state_is_tuple=True)
    # initial_state = lstm_layers.zero_state(None, tf.float32)
    rnn_outputs, _state = tf.nn.dynamic_rnn(cell=lstm_layers, inputs=lookup, dtype=tf.float32)
    # print(lstm_cell)
    # print(lstm_dropout)
    # print(lstm_layers)
    # print('outputs', outputs)
    # print('_state ', _state)
    last_hidden_state = _state[1][1]
    concat = tf.concat([last_hidden_state, X_feat], axis=1)
    print(concat)
    dense1 = tf.layers.dense(concat, dense_size, activation=tf.nn.relu)
    dense2 = tf.layers.dense(dense1, dense_size, activation=tf.nn.relu)
    dense3 = tf.layers.dense(dense2, dense_size, activation=tf.nn.relu)
    logits = tf.layers.dense(dense3, n_classes, activation=None)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # print(dense_output)

    predictions = tf.argmax(logits, 1)
    correct_prediction = tf.equal(predictions, tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver(max_to_keep=10)



tf.reset_default_graph()
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())

    if mode == 'train':
        print('Learning Started')
        for epoch in range(training_epoch):
            i = 0
            print('epoch : {}\n'.format(epoch))
            batch_acc, batch_cost, batch_count = 0.0, 0.0, 1.0

            while i < len(X_train_seq):
                start = i
                end = i + batch_size
                batch_x_seq = np.array(X_train_seq[start:end])
                batch_x_feat = np.array(X_train_feat[start:end])
                batch_y = np.array(y_train[start:end])
                # print(len(batch_x_seq))
                # print(len(batch_x_feat))
                # print(len(batch_y))
                c, acc, _ = sess.run([cost, accuracy, optimizer],
                                     feed_dict={X_seq: batch_x_seq,
                                                X_feat: batch_x_feat,
                                                Y: batch_y,
                                                embedding_placeholder: embedding,
                                                keep_prob: dropout_keep_prob})

                i += batch_size
                batch_acc += acc
                batch_cost += c

                if i % (batch_size * 20) == 0:
                    print('batch count : {}'.format(batch_count))
                    print('batch acc : {}'.format(batch_acc / batch_count))
                    print('batch cost : {}\n'.format(batch_cost / batch_count))
                batch_count += 1


        saver.save(sess, model_path)
        print('Training Finished!')
    elif mode == 'test':
        saver.restore(sess, model_path)
        print('model load finish!')
        # pred, acc = sess.run([predictions, accuracy], feed_dict={X_seq: X_test_seq,
        #                                         X_feat: X_test_feat,
        #                                         Y: y_test,
        #                                         embedding_placeholder: embedding,
        #                                         keep_prob: 1.0})
        pred, acc = sess.run([predictions, accuracy], feed_dict={X_seq: X_test_seq,
                                                                 X_feat: X_test_feat,
                                                                 Y: y_test,
                                                                 embedding_placeholder: embedding,
                                                                 keep_prob: 1.0})
        print('pred :', pred, ', acc :', acc)
        # report_score([LABELS[e] for e in np.argmax(y_test, 1)], [LABELS[e] for e in pred])
        report_score([LABELS[e] for e in np.argmax(y_test, 1)], [LABELS[e] for e in pred])
