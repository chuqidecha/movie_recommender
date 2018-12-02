# -*- coding: utf8 -*-

import logging
import os
import pickle

import tensorflow as tf

from dataset import Dataset, decompression_feature
from inference import full_network, trainable_variable_summaries

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

BATCH_SIZE = 256
EPOCH = 5
DROPOUT_KEEP_PROB = 0.5
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
SHOW_LOG_STEPS = 100
SAVE_MODEL_STEPS = 1000


def train(train_X, train_y, save_dir):
    user_id = tf.placeholder(tf.int32, [None, 1], name='user_id')
    user_gender = tf.placeholder(tf.int32, [None, 1], name='user_gender')
    user_age = tf.placeholder(tf.int32, [None, 1], name='user_age')
    user_job = tf.placeholder(tf.int32, [None, 1], name='user_job')

    movie_id = tf.placeholder(tf.int32, [None, 1], name='movie_id')
    movie_genres = tf.placeholder(tf.float32, [None, 18], name='movie_genres')
    movie_titles = tf.placeholder(tf.int32, [None, 15], name='movie_titles')
    movie_title_length = tf.placeholder(tf.float32, [None], name='movie_title_length')
    targets = tf.placeholder(tf.int32, [None, 1], name='targets')
    dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

    _, _, predicted = full_network(user_id, user_gender, user_age, user_job, movie_id,
                                   movie_genres, movie_titles, movie_title_length,
                                   dropout_keep_prob)

    trainable_variable_summaries()
    with tf.name_scope('loss'):
        # MSE损失，将计算值回归到评分
        loss = tf.losses.mean_squared_error(targets, predicted)
        tf.summary.scalar('loss', loss)

    dataset = Dataset(train_X.values, train_y.values)
    batch_per_epcho = (len(train_X) + BATCH_SIZE - 1) // BATCH_SIZE

    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        batch_per_epcho,
        LEARNING_RATE_DECAY
    )  # 优化损失
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)  # cost

    saver = tf.train.Saver(max_to_keep=(batch_per_epcho * EPOCH + SAVE_MODEL_STEPS - 1) // SAVE_MODEL_STEPS)

    summaries_merged = tf.summary.merge_all()

    with tf.Session() as sess:
        train_summary_dir = os.path.join('./data', 'summaries', 'train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        sess.run(tf.global_variables_initializer())
        for epoch_i in range(EPOCH):
            # 训练的迭代，保存训练损失
            for batch_i in range(batch_per_epcho):
                Xs, ys = dataset.next_batch(BATCH_SIZE)
                users, movies = decompression_feature(Xs)

                feed = {
                    user_id: users.id,
                    user_gender: users.gender,
                    user_age: users.age,
                    user_job: users.job,
                    movie_id: movies.id,
                    movie_genres: movies.genres,
                    movie_titles: movies.titles,
                    movie_title_length: movies.title_length,
                    targets: ys,
                    dropout_keep_prob: DROPOUT_KEEP_PROB}

                step, train_loss, summaries, _ = sess.run([global_step, loss, summaries_merged, train_op], feed)
                train_summary_writer.add_summary(summaries, step)

                if step % SHOW_LOG_STEPS == 0:
                    show_message = 'Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(epoch_i, batch_i,
                                                                                             batch_per_epcho,
                                                                                             train_loss)
                    logging.info(show_message)
                if step % SAVE_MODEL_STEPS == 0:
                    saver.save(sess, save_dir, global_step=global_step)
        saver.save(sess, save_dir, global_step=global_step)


if __name__ == '__main__':
    with open('./data/data.p', 'rb') as data:
        train_X, train_y, _, _ = pickle.load(data, encoding='utf-8')
    train(train_X, train_y, './data/model/model')
