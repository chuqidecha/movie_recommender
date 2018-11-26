# -*- coding: utf-8 -*-
import pickle

import numpy as np
import tensorflow as tf


def get_tensors(loaded_graph):
    uid = loaded_graph.get_tensor_by_name("uid:0")
    user_gender = loaded_graph.get_tensor_by_name("user_gender:0")
    user_age = loaded_graph.get_tensor_by_name("user_age:0")
    user_job = loaded_graph.get_tensor_by_name("user_job:0")
    movie_id = loaded_graph.get_tensor_by_name("movie_id:0")
    movie_categories = loaded_graph.get_tensor_by_name("movie_categories:0")
    movie_titles = loaded_graph.get_tensor_by_name("movie_titles:0")
    targets = loaded_graph.get_tensor_by_name("targets:0")
    dropout_keep_prob = loaded_graph.get_tensor_by_name("dropout_keep_prob:0")

    inference = loaded_graph.get_tensor_by_name(
        "user_movie_fc_layer/user_movie_fc_layer/BiasAdd:0")
    movie_combine_layer_flat = loaded_graph.get_tensor_by_name("movie_fc_layer/Reshape:0")
    user_combine_layer_flat = loaded_graph.get_tensor_by_name("user_fc_layer/Reshape:0")
    return uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, dropout_keep_prob, inference, movie_combine_layer_flat, user_combine_layer_flat

title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data = pickle.load(
    open('./data/preprocess.p', mode='rb'))
movieid2idx = {val[0]: i for i, val in enumerate(movies.values)}

loaded_graph = tf.Graph()  #

user_id_val, movie_id_val = 234, 1401
with tf.Session(graph=loaded_graph) as sess:  #
    # Load saved model
    loader = tf.train.import_meta_graph('./data/save-model.meta')
    loader.restore(sess, './data/save-model')

    # Get Tensors from loaded model
    uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, dropout_keep_prob, inference, _, __ = get_tensors(
        loaded_graph)  # loaded_graph

    categories = np.zeros([1, 18])
    categories[0] = movies.values[movieid2idx[movie_id_val]][2]

    titles = np.zeros([1, 15])
    titles[0] = movies.values[movieid2idx[movie_id_val]][1]

    feed = {
        uid: np.reshape(users.values[user_id_val-1][0], [1, 1]),
        user_gender: np.reshape(users.values[user_id_val-1][1], [1, 1]),
        user_age: np.reshape(users.values[user_id_val-1][2], [1, 1]),
        user_job: np.reshape(users.values[user_id_val-1][3], [1, 1]),
        movie_id: np.reshape(movies.values[movieid2idx[movie_id_val]][0], [1, 1]),
        movie_categories: categories,  #x.take(6,1)
        movie_titles: titles,  #x.take(5,1)
        dropout_keep_prob: 1}

    # Get Prediction
    inference_val = sess.run([inference], feed)
    print(inference_val)