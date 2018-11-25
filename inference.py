# -*- coding: utf-8 -*-
import datetime
import os
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data = pickle.load(
    open('./data/preprocess.p', mode='rb'))
# 嵌入矩阵的维度
embed_dim = 32
# 用户ID个数
uid_max = max(features.take(0, 1)) + 1  # 6040
# 性别个数
gender_count = max(features.take(2, 1)) + 1  # 1 + 1 = 2
# 年龄类别个数
age_count = max(features.take(3, 1)) + 1  # 6 + 1 = 7
# 职业个数
job_count = max(features.take(4, 1)) + 1  # 20 + 1 = 21

# 电影ID个数
movie_id_count = max(features.take(1, 1)) + 1  # 3952
# 电影类型个数
movie_categories_count = max(genres2int.values()) + 1  # 18 + 1 = 19
# 电影名单词个数
movie_title_count = len(title_set)  # 5216

# 电影名长度
sentences_size = 15  # = 15
# 文本卷积滑动窗口，分别滑动2, 3, 4, 5个单词
window_sizes = [3, 5, 7, 9]
# 文本卷积核数量
filter_num = 8

# 电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第5行的数据电影ID不一定是5
movieid2idx = {val[0]: i for i, val in enumerate(movies.values)}

# Number of Epochs
num_epochs = 5
# Batch Size
batch_size = 256

dropout_keep = 0.5
# Learning Rate
learning_rate = 0.0001
# Show stats for every n number of batches
show_every_n_batches = 20

save_dir = './data/save-model'


def user_feature_network(uid, user_gender, user_age, user_job, dropout_keep_prob):
    with tf.variable_scope('uid_embed_layer'):
        uid_embed_matrix = tf.get_variable('uid_embed_matrix', [uid_max, embed_dim],
                                           initializer=tf.truncated_normal_initializer(stddev=0.1))
        uid_embed_layer = tf.nn.embedding_lookup(uid_embed_matrix, uid, name='uid_embed_layer')

    with tf.variable_scope('user_gender_embed_layer'):
        gender_embed_matrix = tf.get_variable('gender_embed_matrix', [gender_count, embed_dim // 2],
                                              initializer=tf.truncated_normal_initializer(stddev=0.1))
        gender_embed_layer = tf.nn.embedding_lookup(gender_embed_matrix, user_gender, name='gender_embed_layer')

    with tf.variable_scope('user_age_embed_layer'):
        age_embed_matrix = tf.get_variable('age_embed_matrix', [age_count, embed_dim // 2],
                                           initializer=tf.truncated_normal_initializer(stddev=0.1))
        age_embed_layer = tf.nn.embedding_lookup(age_embed_matrix, user_age, name='age_embed_layer')

    with tf.variable_scope('user_job_embed_layer'):
        job_embed_matrix = tf.get_variable('job_embed_matrix', [job_count, embed_dim // 2],
                                           initializer=tf.truncated_normal_initializer(stddev=0.1))
        job_embed_layer = tf.nn.embedding_lookup(job_embed_matrix, user_job, name='job_embed_layer')

    uid_fc_layer = tf.layers.dense(uid_embed_layer, embed_dim, name='uid_fc_layer', activation=tf.nn.relu,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
    uid_fc_dropout_layer = tf.layers.dropout(uid_fc_layer, dropout_keep_prob, name='uid_fc_dropout_layer')

    gender_fc_layer = tf.layers.dense(gender_embed_layer, embed_dim, name='gender_fc_layer', activation=tf.nn.relu,
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
    gender_fc_dropout_layer = tf.layers.dropout(gender_fc_layer, dropout_keep_prob, name='gender_fc_dropout_layer')

    age_fc_layer = tf.layers.dense(age_embed_layer, embed_dim, name='age_fc_layer', activation=tf.nn.relu,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
    age_fc_dropout_layer = tf.layers.dropout(age_fc_layer, dropout_keep_prob, name='age_fc_dropout_layer')

    job_fc_layer = tf.layers.dense(job_embed_layer, embed_dim, name='job_fc_layer', activation=tf.nn.relu,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
    job_fc_dropout_layer = tf.layers.dropout(job_fc_layer, dropout_keep_prob, name='job_fc_dropout_layer')

    user_combine_feature = tf.concat(
        [uid_fc_dropout_layer, gender_fc_dropout_layer, age_fc_dropout_layer, job_fc_dropout_layer], 2)
    user_combine_fc_layer = tf.layers.dense(user_combine_feature, 200,
                                            activation=tf.nn.relu,
                                            kernel_regularizer=tf.nn.l2_loss,
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                            name='user_fc_layer')
    user_combine_fc_dropout_layer = tf.layers.dropout(user_combine_fc_layer, dropout_keep_prob,
                                                      name='user_combine_fc_dropout_layer')

    user_combine_layer_flat = tf.reshape(user_combine_fc_dropout_layer, [-1, 200])

    return user_combine_layer_flat


def movie_feature_embedding(movie_id, movie_categories):
    with tf.variable_scope('movie_id_embed_layer'):
        movie_id_embed_matrix = tf.get_variable('movie_id_embed_matrix', [movie_id_count, embed_dim],
                                                initializer=tf.truncated_normal_initializer(stddev=0.1))
        movie_id_embed_layer = tf.nn.embedding_lookup(movie_id_embed_matrix, movie_id, name='movie_id_embed_layer')

    with tf.name_scope('movie_categories_embed_layer'):
        movie_categories_embed_matrix = tf.Variable(tf.random_uniform([movie_categories_count, embed_dim], -1, 1),
                                                    name='movie_categories_embed_matrix')
        movie_categories_embed_layer = tf.nn.embedding_lookup(movie_categories_embed_matrix, movie_categories,
                                                              name='movie_categories_embed_layer')
        movie_categories_embed_layer = tf.reduce_sum(movie_categories_embed_layer, axis=1, keepdims=True)

    return movie_id_embed_layer, movie_categories_embed_layer


def movie_title_cnn_layer(movie_titles):
    with tf.variable_scope('movie_embedding'):
        movie_title_embed_matrix = tf.get_variable('movie_title_embed_matrix', [movie_title_count, embed_dim],
                                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
        movie_title_embed_layer = tf.nn.embedding_lookup(movie_title_embed_matrix, movie_titles,
                                                         name='movie_title_embed_layer')
        movie_title_embed_layer_expand = tf.expand_dims(movie_title_embed_layer, -1)

    pool_layer_lst = []
    for window_size in window_sizes:
        conv_layer = tf.layers.conv2d(movie_title_embed_layer_expand, filter_num,
                                      kernel_size=[window_size, embed_dim],
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      strides=[1, 1],
                                      padding='VALID',
                                      activation=tf.nn.relu,
                                      name='title_cnn_conv_layer_{}'.format(window_size))

        max_pool_layer = tf.layers.max_pooling2d(conv_layer, [sentences_size - window_size + 1, 1], [1, 1],
                                                 padding='VALID',
                                                 name='title_cnn_max_pool_layer_{}'.format(window_size))
        pool_layer_lst.append(max_pool_layer)

    pool_layer = tf.concat(pool_layer_lst, 3, name='title_cnn_pool_layer')
    max_num = len(window_sizes) * filter_num
    pool_layer_flat = tf.reshape(pool_layer, [-1, 1, max_num], name='title_cnn_pool_layer_flat')

    return pool_layer_flat


def movie_feature_network(movie_id_embed_layer, movie_categories_embed_layer, dropout_layer):
    movie_id_fc_layer = tf.layers.dense(movie_id_embed_layer, embed_dim,
                                        activation=tf.nn.relu,
                                        kernel_regularizer=tf.nn.l2_loss,
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                        name="movie_id_fc_layer")
    movie_categories_fc_layer = tf.layers.dense(movie_categories_embed_layer, embed_dim,
                                                activation=tf.nn.relu,
                                                kernel_regularizer=tf.nn.l2_loss,
                                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                name="movie_categories_fc_layer")

    movie_combine_feature = tf.concat([movie_id_fc_layer, movie_categories_fc_layer, dropout_layer], 2)  # (?, 1, 96)
    movie_combine_layer = tf.layers.dense(movie_combine_feature, 200,
                                          activation=tf.nn.relu,
                                          kernel_regularizer=tf.nn.l2_loss,
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                          name="movie_fc_layers")

    movie_combine_layer_flat = tf.reshape(movie_combine_layer, [-1, 200])

    return movie_combine_layer, movie_combine_layer_flat


def train():
    # uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr,dropout_keep_prob
    with tf.variable_scope('input'):
        uid = tf.placeholder(tf.int32, [None, 1], name='uid')
        user_gender = tf.placeholder(tf.int32, [None, 1], name='user_gender')
        user_age = tf.placeholder(tf.int32, [None, 1], name='user_age')
        user_job = tf.placeholder(tf.int32, [None, 1], name='user_job')

        movie_id = tf.placeholder(tf.int32, [None, 1], name='movie_id')
        movie_categories = tf.placeholder(tf.int32, [None, 18], name='movie_categories')
        movie_titles = tf.placeholder(tf.int32, [None, 15], name='movie_titles')
        targets = tf.placeholder(tf.int32, [None, 1], name='targets')
        lr = tf.placeholder(tf.float32, name='learning_rate')
        dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

    # 得到用户特征
    user_combine_layer_flat = user_feature_network(uid, user_gender, user_age, user_job, dropout_keep_prob)
    # 获取电影ID和类别嵌入向量
    movie_id_embed_layer, movie_categories_embed_layer = movie_feature_embedding(movie_id, movie_categories)

    # 获取电影名的特征向量
    pool_layer_flat = movie_title_cnn_layer(movie_titles)  # 得到电影特征
    movie_combine_layer, movie_combine_layer_flat = movie_feature_network(movie_id_embed_layer,
                                                                          movie_categories_embed_layer,
                                                                          pool_layer_flat)

    # 将用户特征和电影特征作为输入，经过全连接，输出一个值的方案
    inference_layer = tf.concat([user_combine_layer_flat, movie_combine_layer_flat], 1)  # (?, 200)
    inference = tf.layers.dense(inference_layer, 1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                kernel_regularizer=tf.nn.l2_loss, name="inference")

    with tf.name_scope("loss"):
        # MSE损失，将计算值回归到评分
        cost = tf.losses.mean_squared_error(targets, inference)
        loss = tf.reduce_mean(cost)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # 优化损失
        train_op = tf.train.AdamOptimizer(lr).minimize(loss,global_step=global_step)  # cost

    with tf.name_scope("params"):
        for variable in tf.trainable_variables():
            name = variable.name.split(':')[0]
            tf.summary.histogram(name, variable)
            mean = tf.reduce_mean(variable)
            tf.summary.scalar("mean/" + name, mean)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(variable - mean)))
            tf.summary.scalar("stddev/" + name, stddev)
    summaries_merged = tf.summary.merge_all()
    saver = tf.train.Saver()

    train_X, test_X, train_y, test_y = train_test_split(features,
                                                        targets_values,
                                                        test_size=0.2,
                                                        random_state=0)

    with tf.Session() as sess:
        train_summary_dir = os.path.join('./data', "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        sess.run(tf.global_variables_initializer())
        for epoch_i in range(num_epochs):
            train_batches = get_batches(train_X, train_y, batch_size)
            test_batches = get_batches(test_X, test_y, batch_size)

            # 训练的迭代，保存训练损失
            for batch_i in range(len(train_X) // batch_size):
                x, y = next(train_batches)

                categories = np.zeros([batch_size, 18])
                for i in range(batch_size):
                    categories[i] = x.take(6, 1)[i]

                titles = np.zeros([batch_size, sentences_size])
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
                    dropout_keep_prob: dropout_keep,  # dropout_keep
                    lr: learning_rate}

                step, train_loss, summaries, _ = sess.run([global_step, loss, summaries_merged, train_op], feed)  # cost
                train_summary_writer.add_summary(summaries, step)  #

                time_str = datetime.datetime.now().isoformat()
                print('{}: Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    time_str,
                    epoch_i,
                    batch_i,
                    (len(train_X) // batch_size),
                    train_loss))


def get_batches(Xs, ys, batch_size):
    for start in range(0, len(Xs), batch_size):
        end = min(start + batch_size, len(Xs))
        yield Xs[start:end], ys[start:end]


if __name__ == '__main__':
    train()
