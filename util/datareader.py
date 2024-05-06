import os
import pickle

import scipy.sparse as sp
import numpy as np
from scipy.io import savemat, loadmat


class Matrix:
    def __init__(self, path, m_type):
        self.train_matrix = None  # 最后形成的训练集矩阵
        self.test_matrix = None  # 最后形成的测试集矩阵
        self.path = path  # 数据文件路径
        self.m_type = m_type  # 文件类型

        self.train_file = os.path.join(self.path, 'train.txt')
        self.test_file = os.path.join(self.path, 'test.txt')
        self.n_users, self.n_items = 0, 0  # 用户数与项目数
        self.train_line, self.test_line = 0, 0  # 训练集与测试集数量

        self.train_dict, self.test_dict = {}, {}

    def count(self):
        """
        该函数统计了用户和项目的数量、训练集与测试集的数量与内容
        """
        max_user, max_item = 0, 0
        train_line_no, test_line_no = 0, 0
        train = open(self.train_file, 'r')
        for line in train.readlines():
            line_content = line.strip().split(' ')

            max_user = max(max_user, int(line_content[0]))
            try:
                max_item = max(max_item, max([int(i) for i in line_content[1:]]))
            except Exception:
                continue
            train_line_no += 1
        train.close()

        test = open(self.test_file, 'r')
        for line in test.readlines():
            line_content = line.strip().split(' ')

            max_user = max(max_user, int(line_content[0]))
            try:
                max_item = max(max_item, max([int(i) for i in line_content[1:]]))
            except Exception:
                continue
            test_line_no += 1
        test.close()

        max_user += 1
        max_item += 1

        self.n_users, self.n_items, self.train_line, self.test_line = \
            max_user, max_item, train_line_no, test_line_no

    def one_to_all(self):
        """
        此类矩阵文件，一行通常包含多个数，第一个数为用户，后续为该用户交互过的项目
        """
        with open(self.train_file) as f_train:
            for line in f_train.readlines():
                if len(line) == 0:
                    break
                line = line.strip('\n')
                items = [int(i) for i in line.split(' ')]
                uid, u_train_items = items[0], items[1:]

                for i in u_train_items:
                    self.train_matrix[uid, i] = 1.
                self.train_dict[uid] = u_train_items

            with open(self.test_file) as f_test:
                for line in f_test.readlines():
                    if len(line) == 0:
                        break
                    line = line.strip('\n')
                    try:
                        items = [int(i) for i in line.split(' ')]
                    except Exception:
                        continue

                    uid, u_test_items = items[0], items[1:]
                    for i in u_test_items:
                        self.test_matrix[uid, i] = 1.
                    self.test_dict[uid] = u_test_items

    def one_to_one(self):
        """
        此类矩阵文件，一行通常包含两到三个数，第一个数为用户，第二个该用户交互过的项目，第三个（如果有）则为评分或交互次数等系数
        """
        with open(self.train_file) as f_train:
            for line in f_train.readlines():
                if len(line) == 0:
                    break
                line = line.strip('\n')
                items = [int(i) for i in line.split(' ')]
                if self.m_type == '1to1':
                    self.train_matrix[items[0], items[1]] = 1.
                else:
                    try:
                        self.train_matrix[items[0], items[1]] = float(items[2])
                    except IndexError:
                        self.train_matrix[items[0], items[1]] = 1.
                try:
                    self.train_dict[items[0]].append(items[1])
                except IndexError:
                    self.train_dict[items[0]] = [items[1]]

            with open(self.test_file) as f_test:
                for line in f_test.readlines():
                    if len(line) == 0:
                        break
                    line = line.strip('\n')
                    try:
                        items = [int(i) for i in line.split(' ')]
                    except Exception:
                        continue
                    try:
                        self.test_matrix[items[0], items[1]] = float(items[2])
                    except IndexError:
                        self.test_matrix[items[0], items[1]] = 1.
                    try:
                        self.test_dict[items[0]].append(items[1])
                    except IndexError:
                        self.test_dict[items[0]] = [items[1]]

    def make_matrix(self):
        """
        根据用户和项目数量，以及文件类型与内容，进行矩阵生成及保存
        """
        self.count()
        self.train_matrix = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.test_matrix = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        if self.m_type == '1toA':
            self.one_to_all()
        elif self.m_type == '1to1' or '1to1_score':
            self.one_to_one()
        else:
            raise "Error: matrix typeError!"

        savemat(os.path.join(self.path, 'matrix.mat'),
                {'users_num': self.n_users, 'items_num': self.n_items,
                 'train_matrix': self.train_matrix, 'test_matrix': self.test_matrix,
                 'train_lines_num': self.train_line, 'test_lines_num': self.test_line})

        with open(os.path.join(self.path, 'train_dict.pkl'), 'wb') as train_dict_file:
            pickle.dump(self.train_dict, train_dict_file)
        with open(os.path.join(self.path, 'test_dict.pkl'), 'wb') as test_dict_file:
            pickle.dump(self.test_dict, test_dict_file)


def read_matrix(data_path='data', matrix_type='1toA'):
    if not os.path.exists(os.path.join(data_path, 'matrix.mat')):
        matrix = Matrix(data_path, matrix_type)
        matrix.make_matrix()

    matrix_info = loadmat(os.path.join(data_path, 'matrix.mat'))
    matrix_info['users_num'] = matrix_info['users_num'].tolist()[0][0]
    matrix_info['items_num'] = matrix_info['items_num'].tolist()[0][0]
    matrix_info['train_lines_num'] = matrix_info['train_lines_num'].tolist()[0][0]
    matrix_info['test_lines_num'] = matrix_info['test_lines_num'].tolist()[0][0]

    with open(os.path.join(data_path, 'train_dict.pkl'), 'rb') as train_dict_file:
        train_dict = pickle.load(train_dict_file)
    with open(os.path.join(data_path, 'test_dict.pkl'), 'rb') as test_dict_file:
        test_dict = pickle.load(test_dict_file)

    return matrix_info, train_dict, test_dict
