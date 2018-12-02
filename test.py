# -*- coding: utf-8 -*-
import logging
import os
import pickle

import tensorflow as tf

from dataset import Dataset, decompression_feature
from inference import full_network

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

BATCH_SIZE = 256
DROPOUT_PROB = 1


def test(test_X, test_y, model_path):
    user_id = tf.placeholder(tf.int32, [None, 1], name='user_id')
    user_gender = tf.placeholder(tf.int32, [None, 1], name='user_gender')
    user_age = tf.placeholder(tf.int32, [None, 1], name='user_age')
    user_job = tf.placeholder(tf.int32, [None, 1], name='user_job')

    movie_id = tf.placeholder(tf.int32, [None, 1], name='movie_id')
    movie_genres = tf.placeholder(tf.float32, [None, 18], name='movie_categories')
    movie_titles = tf.placeholder(tf.int32, [None, 15], name='movie_titles')
    movie_title_length = tf.placeholder(tf.float32, [None], name='movie_title_length')
    targets = tf.placeholder(tf.int32, [None, 1], name='targets')
    dropout_keep_prob = tf.constant(DROPOUT_PROB, dtype=tf.float32, name='dropout_keep_prob')

    _, _, predicted = full_network(user_id, user_gender, user_age, user_job, movie_id,
                                   movie_genres, movie_titles, movie_title_length,
                                   dropout_keep_prob)

    with tf.name_scope('loss'):
        # MSE损失，将计算值回归到评分
        loss = tf.losses.mean_squared_error(targets, predicted)
        tf.summary.scalar('loss', loss)

    dataset = Dataset(test_X.values, test_y.values)
    batch_per_epcho = (len(test_X) + BATCH_SIZE - 1) // BATCH_SIZE

    saver = tf.train.Saver()

    summaries_merged = tf.summary.merge_all()

    with tf.Session() as sess:
        train_summary_dir = os.path.join('./data', 'summaries', 'test')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        sess.run(tf.global_variables_initializer())

        cpkt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, cpkt.model_checkpoint_path)
        avg_loss = 0
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
                targets: ys}

            test_loss, summaries = sess.run([loss, summaries_merged], feed)
            train_summary_writer.add_summary(summaries, batch_i)
            show_message = 'Batch {:>4}/{}   test_loss = {:.3f}'.format(batch_i, batch_per_epcho, test_loss)
            logging.info(show_message)
            avg_loss = avg_loss + test_loss * len(users.id)
        avg_loss = avg_loss / dataset.size
        logging.info('Loss on test is {:.3f}'.format(avg_loss))


if __name__ == '__main__':
    with open('./data/data.p', 'rb') as data:
        _, _, test_X, test_y = pickle.load(data, encoding='utf-8')
    test(test_X, test_y, './data/model')
