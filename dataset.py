# -*- coding: utf-8 -*-
import collections
import pickle

import numpy as np
from sklearn.utils import shuffle


class Dataset(object):
    def __init__(self, Xs, ys=None, shuffle=True):
        self._Xs = Xs
        self._ys = ys
        self._shuffle = True
        self._size = len(self._Xs)
        self._index = list(range(self._size))
        self._current = 0
        self._epoch = 0

    def next_batch(self, batch_size):
        if self._shuffle:
            if self._current >= self._size:
                self._epoch += 1
                self._current = 0
            if self._current == 0:
                self._index = shuffle(self._index)

        start = self._current
        end = min(self._current + batch_size, self._size)
        self._current = end

        Xs = self._Xs[self._index[start:end]]
        if self._ys is not None:
            ys = self._ys[self._index[start:end]]
            return Xs, ys
        else:
            return Xs

    @property
    def epoch(self):
        return self._epoch

    @property
    def size(self):
        return self._size


Users = collections.namedtuple('Users', ['id', 'age', 'gender', 'job'])
Movies = collections.namedtuple('Movies', ['id', 'genres', 'titles', 'title_length'])


def decompression_feature(Xs):
    bath = len(Xs)

    user_id = np.reshape(Xs.take(0, 1), [bath, 1])
    user_gender = np.reshape(Xs.take(5, 1), [bath, 1])
    user_age = np.reshape(Xs.take(6, 1), [bath, 1])
    user_job = np.reshape(Xs.take(4, 1), [bath, 1])
    users = Users(user_id, user_age, user_gender, user_job)

    movie_id = np.reshape(Xs.take(1, 1), [bath, 1])
    movie_genres = np.array(list(Xs.take(10, 1)))
    movie_titles = np.array(list(Xs.take(11, 1)))
    movie_title_length = (movie_titles != 0).sum(axis=1)
    movies = Movies(movie_id, movie_genres, movie_titles, movie_title_length)

    return users, movies


if __name__ == '__main__':
    with open('./data/data.p', 'rb') as data:
        train_X, train_y, _, _ = pickle.load(data, encoding='utf-8')
    dataset = Dataset(train_X.values, train_y.values)

    for i in range(2):
        Xs, ys = dataset.next_batch(2)
        users, movies = decompression_feature(Xs)
        print(users)
        print(movies)
        print(ys)
