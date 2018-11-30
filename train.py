# -*- coding: utf8 -*-
# @Time     : 11/30/18 10:28 PM
# @Author   : yinwb
# @File     : train.py
import datetime
import os
import pickle

import tensorflow as tf

from dataset import Dataset, decompression_feature
from inference import full_network, trainable_variable_summaries

BATCH_SIZE = 256
EPCHO = 5
DROPOUT_PRO = 0.5


def train(train_X, train_y, save_dir):
    user_id = tf.placeholder(tf.int32, [None, 1], name='user_id')
    user_gender = tf.placeholder(tf.int32, [None, 1], name='user_gender')
    user_age = tf.placeholder(tf.int32, [None, 1], name='user_age')
    user_job = tf.placeholder(tf.int32, [None, 1], name='user_job')

    movie_id = tf.placeholder(tf.int32, [None, 1], name='movie_id')
    movie_genres = tf.placeholder(tf.float32, [None, 18], name='movie_categories')
    movie_titles = tf.placeholder(tf.int32, [None, 15], name='movie_titles')
    movies_title_length = tf.placeholder(tf.float32, [None])
    targets = tf.placeholder(tf.int32, [None, 1], name='targets')
    dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

    predicted = full_network(user_id, user_gender, user_age, user_job, movie_id,
                             movie_genres, movie_titles,movies_title_length,
                             dropout_keep_prob)

    trainable_variable_summaries()
    with tf.name_scope('loss'):
        # MSE损失，将计算值回归到评分
        cost = tf.losses.mean_squared_error(targets, predicted)
        loss = tf.reduce_mean(cost)
        tf.summary.scalar('loss', loss)

    dataset = Dataset(train_X.values, train_y.values)
    batch_per_epcho = (len(train_X) + BATCH_SIZE - 1) // BATCH_SIZE

    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(
        0.01,
        global_step,
        batch_per_epcho,
        0.99
    )  # 优化损失
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)  # cost

    saver = tf.train.Saver(max_to_keep=EPCHO)

    summaries_merged = tf.summary.merge_all()

    with tf.Session() as sess:
        train_summary_dir = os.path.join('./data', 'summaries', 'train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        sess.run(tf.global_variables_initializer())
        for epoch_i in range(EPCHO):
            # 训练的迭代，保存训练损失
            for batch_i in range(batch_per_epcho):
                Xs, ys = dataset.next_batch(BATCH_SIZE)
                users, movies, ys = decompression_feature(Xs, ys)

                feed = {
                    user_id: users.id,
                    user_gender: users.gender,
                    user_age: users.age,
                    user_job: users.job,
                    movie_id: movies.id,
                    movie_genres: movies.genres,
                    movie_titles: movies.titles,
                    movies_title_length: movies.title_length,
                    targets: ys,
                    dropout_keep_prob: DROPOUT_PRO}  # dropout_keep

                step, train_loss, summaries, _ = sess.run([global_step, loss, summaries_merged, train_op], feed)  # cost
                train_summary_writer.add_summary(summaries, step)

                time_str = datetime.datetime.now().isoformat()
                print('{}: Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    time_str,
                    epoch_i,
                    batch_i,
                    (len(train_X) // BATCH_SIZE),
                    train_loss))
            saver.save(sess, save_dir)


if __name__ == '__main__':
    title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data = pickle.load(
        open('./data/preprocess.p', mode='rb'))
    with open('./data/data.p', 'rb') as data:
        train_X, train_y, _, _ = pickle.load(data, encoding='utf-8')
    train(train_X, train_y, './data/mode')
