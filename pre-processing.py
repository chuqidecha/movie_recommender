# -*- coding: utf-8 -*-

import hashlib
import os
import pickle
import re
import zipfile
from urllib.request import urlretrieve

import pandas as pd
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


dataset_zip = download_ml_1m('./data')


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
            users['Gender'] = users['Gender'].map(gender_map)

            age_map = {val: ii for ii, val in enumerate(set(users['Age']))}
            users['Age'] = users['Age'].map(age_map)

        # 读取Movie数据集
        with zf.open('ml-1m/movies.dat') as movies_raw_data:
            movies_title = ['MovieID', 'Title', 'Genres']
            movies = pd.read_table(movies_raw_data, sep=b'::', header=None, names=movies_title, engine='python')
            movies_orig = movies.values
            # 将Title中的年份去掉
            pattern = re.compile(b'^(.*)\((\d+)\)$')

            title_map = {val: pattern.match(val).group(1) for ii, val in enumerate(set(movies['Title']))}
            movies['Title'] = movies['Title'].map(title_map)

            # 电影类型转数字字典
            genres_set = set()
            for val in movies['Genres'].str.split(b'|'):
                genres_set.update(val)

            genres_set.add('<PAD>')
            genres2int = {val: ii for ii, val in enumerate(genres_set)}

            # 将电影类型转成等长数字列表，长度是18
            genres_map = {val: [genres2int[row] for row in val.split(b'|')] for ii, val in
                          enumerate(set(movies['Genres']))}

            for key in genres_map:
                for cnt in range(max(genres2int.values()) - len(genres_map[key])):
                    genres_map[key].insert(len(genres_map[key]) + cnt, genres2int['<PAD>'])

            movies['Genres'] = movies['Genres'].map(genres_map)

            # 电影Title转数字字典
            title_set = set()
            for val in movies['Title'].str.split():
                title_set.update(val)

            title_set.add('<PAD>')
            title2int = {val: ii for ii, val in enumerate(title_set)}

            # 将电影Title转成等长数字列表，长度是15
            title_count = 15
            title_map = {val: [title2int[row] for row in val.split()] for ii, val in enumerate(set(movies['Title']))}

            for key in title_map:
                for cnt in range(title_count - len(title_map[key])):
                    title_map[key].insert(len(title_map[key]) + cnt, title2int['<PAD>'])

            movies['Title'] = movies['Title'].map(title_map)

        # 读取评分数据集
        with zf.open('ml-1m/ratings.dat') as ratings_raw_data:
            ratings_title = ['UserID', 'MovieID', 'ratings', 'timestamps']
            ratings = pd.read_table(ratings_raw_data, sep=b'::', header=None, names=ratings_title, engine='python')
            ratings = ratings.filter(regex='UserID|MovieID|ratings')

    # 合并三个表
    data = pd.merge(pd.merge(ratings, users), movies)

    # 将数据分成X和y两张表
    target_fields = ['ratings']
    features_pd, targets_pd = data.drop(target_fields, axis=1), data[target_fields]

    features = features_pd.values
    targets_values = targets_pd.values

    return title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data


title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data = load_data(dataset_zip)

pickle.dump((title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data),
            open('./data/preprocess.p', 'wb'))
#
# title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = pickle.load(
#     open('preprocess.p', mode='rb'))
