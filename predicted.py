# -*- coding: utf-8 -*-
import pickle

import numpy as np


def relu(x):
    s = np.where(x < 0, 0, x)
    return s


def predict_rating(user_feature, movie_feature, kernel, bais, activate):
    """
    评分函数
    :param user_feature:
    :param movie_feature:
    :param kernel:
    :param bais:
    :param activate:
    :return:
    """
    feature = np.concatenate((user_feature, movie_feature))
    xw_b = np.dot(feature, kernel) + bais
    output = activate(xw_b)
    return output


def cosine_similiarity(vec_left, vec_right):
    """
    余弦相似度
    :param vec_left: 
    :param vec_right: 
    :return: 
    """
    num = np.dot(vec_left, vec_right)
    denom = np.linalg.norm(vec_left) * np.linalg.norm(vec_right)
    cos = -1 if denom == 0 else num / denom
    return cos


def similar_movie(movie_id, top_k, movie_features):
    """
    相似电影
    :param movie_id: 
    :param top_k: 
    :param movie_features: 
    :return: 
    """
    cosine_similiarities = {}
    movie_feature = movie_features[movie_id]
    for (movie_id_, movie_feature_) in movie_features.items():
        cosine_similiarities[movie_id_] = cosine_similiarity(movie_feature, movie_feature_)
    return sorted(cosine_similiarities.items(), key=lambda item: item[1])[-top_k:]


def similar_user(user_id, top_k, user_features):
    """
    相似用户
    :param user_id: 
    :param top_k: 
    :param user_features: 
    :return: 
    """
    cosine_similiarities = {}
    user_feature = user_features[user_id]
    for (user_id_, user_feature_) in user_features.items():
        cosine_similiarities[user_id_] = cosine_similiarity(user_feature, user_feature_)
    return sorted(cosine_similiarities.items(), key=lambda item: item[1])[-top_k:]


if __name__ == '__main__':
    with open('./data/user-features.p', 'rb') as uf:
        user_features = pickle.load(uf, encoding='utf-8')

    with open('./data/movie-features.p', 'rb') as mf:
        movie_features = pickle.load(mf)

    with open('./data/user-movie-fc-param.p', 'rb') as params:
        kernel, bais = pickle.load(params, encoding='utf-8')

    with open('./data/movies.p', 'rb') as mv:
        movies = pickle.load(mv, encoding='utf-8')
    with open('./data/users.p', 'rb') as usr:
        users = pickle.load(usr, encoding='utf-8')

    rating1 = predict_rating(user_features[1], movie_features[1193], kernel, bais, relu)
    print('UserID={:>4},MovieID={:>4},Rating={:.3f}'.format(1, 1193, rating1[0]))
    rating2 = predict_rating(user_features[5900], movie_features[3100], kernel, bais, relu)
    print('UserID={:>4},MovieID={:>4},Rating={:.3f}'.format(234, 1401, rating2[0]))

    similar_users = similar_user(5900, 5, user_features)
    print('These Users are similar to {}'.format(str(users[users['UserID'] == 1642].to_dict('records'))))
    for user in similar_users:
        print(users[users['UserID'] == user[0]].to_dict('records')[0])

    similar_movies = similar_movie(1401, 5, movie_features)
    print('These Movie are similar to {}'.format(
        str(movies[movies['MovieID'] == 1401][['MovieID', 'Title', 'Genres']].to_dict('records'))))
    for movie in similar_movies:
        print(movies[movies['MovieID'] == movie[0]][['MovieID', 'Title', 'Genres']].to_dict('records')[0])
