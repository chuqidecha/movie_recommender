# -*- coding: utf-8 -*-

import logging
import pickle

import numpy as np
import tensorflow as tf

from dataset import Dataset
from inference import full_network

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

BATCH_SIZE = 256
DROPOUT_PROB = 1


def main(model_path):
    user_id = tf.placeholder(tf.int32, [None, 1], name='user_id')
    user_gender = tf.placeholder(tf.int32, [None, 1], name='user_gender')
    user_age = tf.placeholder(tf.int32, [None, 1], name='user_age')
    user_job = tf.placeholder(tf.int32, [None, 1], name='user_job')

    movie_id = tf.placeholder(tf.int32, [None, 1], name='movie_id')
    movie_genres = tf.placeholder(tf.float32, [None, 18], name='movie_categories')
    movie_titles = tf.placeholder(tf.int32, [None, 15], name='movie_titles')
    movie_title_length = tf.placeholder(tf.float32, [None], name='movie_title_length')
    dropout_keep_prob = tf.constant(DROPOUT_PROB, dtype=tf.float32, name='dropout_keep_prob')

    user_feature, movie_feature, _ = full_network(user_id, user_gender, user_age, user_job, movie_id,
                                                  movie_genres, movie_titles, movie_title_length,
                                                  dropout_keep_prob)

    with tf.variable_scope('user_movie_fc', reuse=True):
        user_movie_fc_kernel = tf.get_variable('kernel')
        user_movie_fc_bias = tf.get_variable('bias')

    with open('./data/users.p', 'rb') as users:
        user_Xs = pickle.load(users)
    with open('./data/movies.p', 'rb') as movies:
        movie_Xs = pickle.load(movies)

    user_dataset = Dataset(user_Xs.values, shuffle=False)
    movie_dataset = Dataset(movie_Xs.values, shuffle=False)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        cpkt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, cpkt.model_checkpoint_path)

        # 提取用户特征
        user_features = {}
        for batch in range((user_dataset.size + BATCH_SIZE - 1) // BATCH_SIZE):
            data = user_dataset.next_batch(BATCH_SIZE)
            feed = {
                user_id: np.reshape(data.take(0, 1), [len(data), 1]),
                user_gender: np.reshape(data.take(4, 1), [len(data), 1]),
                user_age: np.reshape(data.take(5, 1), [len(data), 1]),
                user_job: np.reshape(data.take(3, 1), [len(data), 1]),
            }
            feature = sess.run(user_feature, feed_dict=feed)
            user_features.update({key: value for (key, value) in zip(data.take(0, 1), feature)})
        with open('./data/user-features.p', 'wb') as uf:
            pickle.dump(user_features, uf)

        # 提取电影特征
        movie_features = {}
        for batch in range((movie_dataset.size + BATCH_SIZE - 1) // BATCH_SIZE):
            data = movie_dataset.next_batch(BATCH_SIZE)
            feed = {
                movie_id: np.reshape(data.take(0, 1), [len(data), 1]),
                movie_genres: np.array(list(data.take(4, 1))),
                movie_titles: np.array(list(data.take(5, 1))),
                movie_title_length: (np.array(list(data.take(5, 1))) != 0).sum(axis=1)
            }
            feature = sess.run(movie_feature, feed_dict=feed)
            movie_features.update({key: value for (key, value) in zip(data.take(0, 1), feature)})
        with open('./data/movie-features.p', 'wb') as mf:
            pickle.dump(movie_features, mf)

        # 保存损失层的kenel和biase
        kernel, bais = sess.run([user_movie_fc_kernel, user_movie_fc_bias])
        with open('./data/user-movie-fc-param.p', 'wb') as params:
            pickle.dump((kernel, bais), params)


if __name__ == '__main__':
    main('./data/model')
