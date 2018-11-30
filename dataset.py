# -*- coding: utf-8 -*-
import pickle

from sklearn.utils import shuffle


class Dataset(object):
    def __init__(self, Xs, ys, shuffle=True):
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

        end = min(self._current + batch_size, self._size)
        Xs = self._Xs[self._index[self._current:end]]
        ys = self._ys[self._index[self._current:end]]
        self._current = end
        return Xs, ys

    @property
    def epoch(self):
        return self._epoch



if __name__ == '__main__':
    with open('./data/data.p', 'rb') as data:
        train_X, train_y, _, _ = pickle.load(data, encoding='utf-8')
    dataset = Dataset(train_X.values, train_y.values)
    print(dataset.next_batch(10))
