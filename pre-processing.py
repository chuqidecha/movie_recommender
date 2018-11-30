# -*- coding: utf-8 -*-

import hashlib
import os
import pickle
import re
import zipfile
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class DLProgress(tqdm):
    """
    Handle Progress Bar while Downloading
    """
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        """
        A hook function that will be called once on establishment of the network connection and
        once after each block read thereafter.
        :param block_num: A count of blocks transferred so far
        :param block_size: Block size in bytes
        :param total_size: The total size of the file. This may be -1 on older FTP servers which do not return
                            a file size in response to a retrieval request.
        """
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def download_ml_1m(save_path):
    """
    Download and extract database
    :param database_name: Database name
    """

    url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
    hash_code = 'c4d9eecfca2ab87c1945afe126590906'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_pathname = os.path.join(save_path, 'ml-1m.zip')

    if os.path.exists(save_pathname):
        print('skip download ml-1m.zip because the file exists !')
    else:
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Downloading ml-1m') as pbar:
            urlretrieve(
                url,
                save_pathname,
                pbar.hook)

        assert hashlib.md5(open(save_pathname, 'rb').read()).hexdigest() == hash_code, \
            '{} file is corrupted. Remove the file and try again.'.format(save_path)

        print('Download ml-1m.zip successfully !')

    return save_pathname


def genres_multi_hot(genre_int_map):
    '''
    电影类型使用multi-hot编码
    :param genre_int_map:
    :return:
    '''

    def helper(genres):
        genre_int_list = [genre_int_map[genre] for genre in genres.split(b'|')]
        multi_hot = np.zeros(len(genre_int_map))
        multi_hot[genre_int_list] = 1
        return multi_hot

    return helper


def title_encode(words_int_map):
    '''
    将电影Title转成长度为15的数字列表，如果长度小于15则用0填充，大于15则截断
    :param words_int_map:
    :return:
    '''

    def helper(title):
        title_words = [words_int_map[word] for word in title.split()]
        if len(title_words) > 15:
            return np.array(title[:15])
        else:
            title_vector = np.zeros(15)
            title_vector[:len(title_words)] = title_words
            return title_vector

    return helper


def load_data(dataset_zip):
    """
    Load Dataset from Zip File
    """
    with zipfile.ZipFile(dataset_zip) as zf:
        # 读取User数据
        with zf.open('ml-1m/users.dat') as users_raw_data:
            users_title = ['UserID', 'Gender', 'Age', 'JobID', 'Zip-code']
            users = pd.read_table(users_raw_data, sep=b'::', header=None, names=users_title, engine='python')
            users = users.filter(regex='UserID|Gender|Age|JobID')

            # 改变User数据中性别和年龄
            gender_map = {b'F': 0, b'M': 1}
            users['GenderIndex'] = users['Gender'].map(gender_map)

            age_map = {val: ii for ii, val in enumerate(set(users['Age']))}
            users['AgeIndex'] = users['Age'].map(age_map)

        # 读取Movie数据集
        with zf.open('ml-1m/movies.dat') as movies_raw_data:
            movies_title = ['MovieID', 'Title', 'Genres']
            movies = pd.read_table(movies_raw_data, sep=b'::', header=None, names=movies_title, engine='python')
            # 将Title中的年份去掉
            pattern = re.compile(b'^(.*)\((\d+)\)$')

            movies['TitleWithoutYear'] = movies['Title'].map(lambda x: pattern.match(x).group(1))
            # 电影类型转数字字典
            genres_set = set()
            for val in movies['Genres'].str.split(b'|'):
                genres_set.update(val)

            genre_int_map = {val: ii for ii, val in enumerate(genres_set)}

            movies['GenresMultiHot'] = movies['Genres'].map(genres_multi_hot(genre_int_map))

            # 电影Title转数字字典
            words_set = set()
            for val in movies['TitleWithoutYear'].str.split():
                words_set.update(val)

            words_int_map = {val: ii for ii, val in enumerate(words_set, start=1)}

            movies['TitleIndex'] = movies['TitleWithoutYear'].map(title_encode(words_int_map))

        # 读取评分数据集
        with zf.open('ml-1m/ratings.dat') as ratings_raw_data:
            ratings_title = ['UserID', 'MovieID', 'ratings', 'timestamps']
            ratings = pd.read_table(ratings_raw_data, sep=b'::', header=None, names=ratings_title, engine='python')
            ratings = ratings.filter(regex='UserID|MovieID|ratings')

    # 合并三个表
    data = pd.merge(pd.merge(ratings, users), movies)

    # 将数据分成X和y两张表
    features, targets = data.drop(['ratings'], axis=1), data[['ratings']]

    return features, targets, age_map, gender_map, genre_int_map, words_int_map


if __name__ == '__main__':
    dataset_zip = download_ml_1m('./data')
    features, targets, age_map, gender_map, genre_int_map, words_int_map = load_data(dataset_zip)

    with open('./data/meta.p', 'wb') as meta:
        pickle.dump((age_map, gender_map, genre_int_map, words_int_map), meta)

    train_X, test_X, train_y, test_y = train_test_split(features, targets, test_size=0.2, random_state=0)
    with open('./data/data.p', 'wb') as data:
        pickle.dump((train_X, train_y, test_X, test_y), data)
