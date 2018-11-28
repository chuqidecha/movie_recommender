# -*- coding: utf-8 -*-
import datetime
import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data = pickle.load(
#     open('./data/preprocess.p', mode='rb'))
# 嵌入矩阵的维度
EMBED_DIM = 32
# 用户ID个数
UID_COUNT = 6040
# 性别个数
GENDER_COUNT = 2
# 年龄类别个数
AGE_COUNT = 7
# 职业个数
JOB_COUNT = 21

# 电影ID个数
MOVIE_ID_COUNT = 3952
# 电影类型个数
MOVIE_CATEGORIES_COUNT = 19
# 电影名单词个数
MOVIE_TITLE_COUNT = 5217
# 电影名长度
MOVIE_TITILE_WORD_COUNT = 15
# 文本卷积滑动窗口，分别滑动2, 3, 4, 5个单词
CNN_WINDOW_SIZES = [3, 5, 7, 9]
# 文本卷积核数量
CNN_FILTER_COUNT = 8

# 电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第5行的数据电影ID不一定是5
# movieid2idx = {val[0]: i for i, val in enumerate(movies.values)}

# Number of Epochs
num_epochs = 10
# Batch Size
batch_size = 256

DROPOUT_KEEP = 0.5
# Learning Rate
learning_rate_base = 0.0001
# Show stats for every n number of batches
show_every_n_batches = 20

save_dir = './data/save-model/recommender'


def user_feature_network(uid, user_gender, user_age, user_job, dropout_keep_prob):
    with tf.variable_scope('uid_embed_layer'):
        uid_embed_matrix = tf.get_variable('uid_embed_matrix', [UID_COUNT, EMBED_DIM],
                                           initializer=tf.truncated_normal_initializer(stddev=0.1))
        uid_embed_layer = tf.nn.embedding_lookup(uid_embed_matrix, uid, name='uid_embed_layer')

    with tf.variable_scope('user_gender_embed_layer'):
        gender_embed_matrix = tf.get_variable('gender_embed_matrix', [GENDER_COUNT, EMBED_DIM // 2],
                                              initializer=tf.truncated_normal_initializer(stddev=0.1))
        gender_embed_layer = tf.nn.embedding_lookup(gender_embed_matrix, user_gender, name='gender_embed_layer')

    with tf.variable_scope('user_age_embed_layer'):
        age_embed_matrix = tf.get_variable('age_embed_matrix', [AGE_COUNT, EMBED_DIM // 2],
                                           initializer=tf.truncated_normal_initializer(stddev=0.1))
        age_embed_layer = tf.nn.embedding_lookup(age_embed_matrix, user_age, name='age_embed_layer')

    with tf.variable_scope('user_job_embed_layer'):
        job_embed_matrix = tf.get_variable('job_embed_matrix', [JOB_COUNT, EMBED_DIM // 2],
                                           initializer=tf.truncated_normal_initializer(stddev=0.1))
        job_embed_layer = tf.nn.embedding_lookup(job_embed_matrix, user_job, name='job_embed_layer')

    uid_fc_layer = tf.layers.dense(uid_embed_layer, EMBED_DIM,
                                   activation=tf.nn.relu,
                                   kernel_regularizer=tf.nn.l2_loss,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                   name='uid_fc_layer')
    uid_fc_dropout_layer = tf.layers.dropout(uid_fc_layer, dropout_keep_prob, name='uid_fc_dropout_layer')

    gender_fc_layer = tf.layers.dense(gender_embed_layer, EMBED_DIM,
                                      activation=tf.nn.relu,
                                      kernel_regularizer=tf.nn.l2_loss,
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      name='gender_fc_layer')
    gender_fc_dropout_layer = tf.layers.dropout(gender_fc_layer, dropout_keep_prob, name='gender_fc_dropout_layer')

    age_fc_layer = tf.layers.dense(age_embed_layer, EMBED_DIM,
                                   activation=tf.nn.relu,
                                   kernel_regularizer=tf.nn.l2_loss,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                   name='age_fc_layer')
    age_fc_dropout_layer = tf.layers.dropout(age_fc_layer, dropout_keep_prob, name='age_fc_dropout_layer')

    job_fc_layer = tf.layers.dense(job_embed_layer, EMBED_DIM,
                                   activation=tf.nn.relu,
                                   kernel_regularizer=tf.nn.l2_loss,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                   name='job_fc_layer')
    job_fc_dropout_layer = tf.layers.dropout(job_fc_layer, dropout_keep_prob, name='job_fc_dropout_layer')

    with tf.name_scope('user_fc_layer'):
        user_combine_feature = tf.concat(
            [uid_fc_dropout_layer, gender_fc_dropout_layer, age_fc_dropout_layer, job_fc_dropout_layer], 2)
        user_combine_fc_layer = tf.layers.dense(user_combine_feature, 200,
                                                activation=tf.nn.relu,
                                                kernel_regularizer=tf.nn.l2_loss,
                                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                name='user_fc_layer')
        user_combine_layer_flat = tf.reshape(user_combine_fc_layer, [-1, 200])

    return user_combine_layer_flat


def movie_feature_embed_network(movie_id, movie_categories):
    with tf.variable_scope('movie_id_embed_layer'):
        movie_id_embed_matrix = tf.get_variable('movie_id_embed_matrix', [MOVIE_ID_COUNT, EMBED_DIM],
                                                initializer=tf.truncated_normal_initializer(stddev=0.1))
        movie_id_embed_layer = tf.nn.embedding_lookup(movie_id_embed_matrix, movie_id, name='movie_id_embed_layer')

    with tf.name_scope('movie_categories_embed_layer'):
        movie_categories_embed_matrix = tf.Variable(tf.random_uniform([MOVIE_CATEGORIES_COUNT, EMBED_DIM], -1, 1),
                                                    name='movie_categories_embed_matrix')
        movie_categories_embed_layer = tf.nn.embedding_lookup(movie_categories_embed_matrix, movie_categories,
                                                              name='movie_categories_embed_layer')
        movie_categories_embed_layer = tf.reduce_sum(movie_categories_embed_layer, axis=1, keepdims=True)

    return movie_id_embed_layer, movie_categories_embed_layer


def movie_title_lstm_layer(movie_titles, dropout_keep_prob):
    with tf.variable_scope('movie_title_embed_layer'):
        movie_title_embed_matrix = tf.get_variable('movie_title_embed_matrix', [MOVIE_TITLE_COUNT, EMBED_DIM],
                                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
        movie_title_embed_layer = tf.nn.embedding_lookup(movie_title_embed_matrix, movie_titles,
                                                         name='movie_title_embed_layer')
        # movie_title_embed_layer_expand = tf.expand_dims(movie_title_embed_layer, -1)

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(128, forget_bias=0.0)
    lstm_cell_dropout = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, dropout_keep_prob)
    batch_size_ = tf.shape(movie_titles)[0]
    init_state = lstm_cell_dropout.zero_state(batch_size_, dtype=tf.float32)
    output, final_state = tf.nn.dynamic_rnn(lstm_cell_dropout, movie_title_embed_layer, initial_state=init_state)
    output = tf.reduce_sum(output, 1)
    return output


def movie_title_cnn_layer(movie_titles):
    with tf.variable_scope('movie_title_embed_layer'):
        movie_title_embed_matrix = tf.get_variable('movie_title_embed_matrix', [MOVIE_TITLE_COUNT, EMBED_DIM],
                                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
        movie_title_embed_layer = tf.nn.embedding_lookup(movie_title_embed_matrix, movie_titles,
                                                         name='movie_title_embed_layer')
        movie_title_embed_layer_expand = tf.expand_dims(movie_title_embed_layer, -1)

    pool_layer_lst = []
    for window_size in CNN_WINDOW_SIZES:
        conv_layer = tf.layers.conv2d(movie_title_embed_layer_expand, CNN_FILTER_COUNT,
                                      kernel_size=[window_size, EMBED_DIM],
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      strides=[1, 1],
                                      padding='VALID',
                                      activation=tf.nn.relu,
                                      name='movie_title_cnn_conv_layer_{}'.format(window_size))

        max_pool_layer = tf.layers.max_pooling2d(conv_layer, [MOVIE_TITILE_WORD_COUNT - window_size + 1, 1], [1, 1],
                                                 padding='VALID',
                                                 name='movie_title_cnn_max_pool_layer_{}'.format(window_size))
        pool_layer_lst.append(max_pool_layer)

    with tf.name_scope('movie_title_output_layer'):
        pool_layer = tf.concat(pool_layer_lst, 3)
        max_num = len(CNN_WINDOW_SIZES) * CNN_FILTER_COUNT
        pool_layer_flat = tf.reshape(pool_layer, [-1, 1, max_num])

    return pool_layer_flat


def movie_feature_network(movie_id_embed_layer, movie_categories_embed_layer, movie_title_output_layer,
                          dropout_keep_prob):
    movie_id_fc_layer = tf.layers.dense(movie_id_embed_layer, EMBED_DIM,
                                        activation=tf.nn.relu,
                                        kernel_regularizer=tf.nn.l2_loss,
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                        name='movie_id_fc_layer')
    movie_id_dropout_layer = tf.layers.dropout(movie_id_fc_layer, dropout_keep_prob, name='movie_id_dropout_layer')

    movie_categories_fc_layer = tf.layers.dense(movie_categories_embed_layer, EMBED_DIM,
                                                activation=tf.nn.relu,
                                                kernel_regularizer=tf.nn.l2_loss,
                                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                name='movie_categories_fc_layer')
    movie_categories_dropout_layer = tf.layers.dropout(movie_categories_fc_layer, dropout_keep_prob,
                                                       name='movie_categories_dropout_layer')

    with tf.name_scope('movie_fc_layer'):
        movie_combine_feature = tf.concat(
            [movie_id_dropout_layer, movie_categories_dropout_layer, movie_title_output_layer], 2)
        movie_combine_layer = tf.layers.dense(movie_combine_feature, 200,
                                              activation=tf.nn.relu,
                                              kernel_regularizer=tf.nn.l2_loss,
                                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                              name='movie_fc_layer')
        movie_combine_layer_flat = tf.reshape(movie_combine_layer, [-1, 200])

    return movie_combine_layer_flat


def full_network(uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, dropout_keep_prob):
    # 得到用户特征
    user_combine_layer_flat = user_feature_network(uid, user_gender, user_age, user_job, dropout_keep_prob)
    # 获取电影ID和类别嵌入向量
    movie_id_embed_layer, movie_categories_embed_layer = movie_feature_embed_network(movie_id, movie_categories)

    # 获取电影名的特征向量
    pool_layer_flat = movie_title_lstm_layer(movie_titles,dropout_keep_prob)  # 得到电影特征
    movie_combine_layer_flat = movie_feature_network(movie_id_embed_layer,
                                                     movie_categories_embed_layer,
                                                     pool_layer_flat,
                                                     dropout_keep_prob)

    # 将用户特征和电影特征作为输入，经过全连接，输出一个值
    with tf.name_scope('user_movie_fc_layer'):
        input_layer = tf.concat([user_combine_layer_flat, movie_combine_layer_flat], 1)  # (?, 200)
        predicted = tf.layers.dense(input_layer, 1,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    kernel_regularizer=tf.nn.l2_loss,
                                    name='user_movie_fc_layer')
    return predicted


def trainable_variable_summaries():
    for variable in tf.trainable_variables():
        name = variable.name.split(':')[0]
        tf.summary.histogram(name, variable)
        mean = tf.reduce_mean(variable)
        tf.summary.scalar('mean/' + name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(variable - mean)))
        tf.summary.scalar('stddev/' + name, stddev)


def train(train_X, train_y):
    uid = tf.placeholder(tf.int32, [None, 1], name='uid')
    user_gender = tf.placeholder(tf.int32, [None, 1], name='user_gender')
    user_age = tf.placeholder(tf.int32, [None, 1], name='user_age')
    user_job = tf.placeholder(tf.int32, [None, 1], name='user_job')

    movie_id = tf.placeholder(tf.int32, [None, 1], name='movie_id')
    movie_categories = tf.placeholder(tf.int32, [None, 18], name='movie_categories')
    movie_titles = tf.placeholder(tf.int32, [None, 15], name='movie_titles')
    targets = tf.placeholder(tf.int32, [None, 1], name='targets')
    dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

    predicted = full_network(uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles,
                             dropout_keep_prob)
    summaries_merged = tf.summary.merge_all()
    with tf.name_scope('loss'):
        # MSE损失，将计算值回归到评分
        cost = tf.losses.mean_squared_error(targets, predicted)
        loss = tf.reduce_mean(cost)
        tf.summary.scalar('loss', loss)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(
        learning_rate_base,
        global_step,
        len(train_X) // batch_size,
        0.99
    )  # 优化损失
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)  # cost

    saver = tf.train.Saver(max_to_keep=num_epochs)

    with tf.Session() as sess:
        train_summary_dir = os.path.join('./data', 'summaries', 'train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        sess.run(tf.global_variables_initializer())
        for epoch_i in range(num_epochs):
            train_batches = get_batches(train_X, train_y, batch_size)

            # 训练的迭代，保存训练损失
            for batch_i in range(len(train_X) // batch_size):
                x, y = next(train_batches)

                categories = np.zeros([batch_size, 18])
                for i in range(batch_size):
                    categories[i] = x.take(6, 1)[i]

                titles = np.zeros([batch_size, MOVIE_TITILE_WORD_COUNT])
                for i in range(batch_size):
                    titles[i] = x.take(5, 1)[i]

                feed = {
                    uid: np.reshape(x.take(0, 1), [batch_size, 1]),
                    user_gender: np.reshape(x.take(2, 1), [batch_size, 1]),
                    user_age: np.reshape(x.take(3, 1), [batch_size, 1]),
                    user_job: np.reshape(x.take(4, 1), [batch_size, 1]),
                    movie_id: np.reshape(x.take(1, 1), [batch_size, 1]),
                    movie_categories: categories,  # x.take(6,1)
                    movie_titles: titles,  # x.take(5,1)
                    targets: np.reshape(y, [batch_size, 1]),
                    dropout_keep_prob: DROPOUT_KEEP}  # dropout_keep

                step, train_loss, summaries, _ = sess.run([global_step, loss, summaries_merged, train_op], feed)  # cost
                train_summary_writer.add_summary(summaries, step)  #

                time_str = datetime.datetime.now().isoformat()
                print('{}: Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    time_str,
                    epoch_i,
                    batch_i,
                    (len(train_X) // batch_size),
                    train_loss))
            saver.save(sess, save_dir)


def get_batches(Xs, ys, batch_size):
    for start in range(0, len(Xs), batch_size):
        end = min(start + batch_size, len(Xs))
        yield Xs[start:end], ys[start:end]


if __name__ == '__main__':
    title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data = pickle.load(
        open('./data/preprocess.p', mode='rb'))
    train_X, test_X, train_y, test_y = train_test_split(features,
                                                        targets_values,
                                                        test_size=0.2,
                                                        random_state=0)
    train(train_X,train_y)


