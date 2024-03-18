# import logging
# import numpy as np
# import os
# import pickle
# import scipy.sparse as sp
# import sys
# import tensorflow as tf
# from sklearn.preprocessing import MinMaxScaler
# from scipy.sparse import linalg
# import math
# import torch
# from torch.utils.data import Dataset, DataLoader
#
#
# class CustomDataset(Dataset):
#     def __init__(self, dataset_dir, category):
#         cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
#         self.features = torch.tensor(cat_data['x'], dtype=torch.float32)
#
#         self.labels = torch.tensor(cat_data['y'], dtype=torch.float32)
#
#     def __len__(self):
#         return len(self.features)
#
#     def __getitem__(self, idx):
#
#         return self.features[idx], self.labels[idx]
#
#
# class CustomDataLoader(DataLoader):
#     def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
#                  batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False,
#                  drop_last=False, timeout=0, worker_init_fn=None):
#         super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers,
#                          collate_fn, pin_memory, drop_last, timeout, worker_init_fn)
#
#         self.size = len(dataset)
#         self.num_batch = math.ceil(self.size / self.batch_size)
#
# # class DataLoader(object):
# #     def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
# #         """
# #
# #         :param xs:
# #         :param ys:
# #         :param batch_size:
# #         :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
# #                                      用最后一个样本填充，使样本数量可被batch_size整除
# #         """
# #         self.batch_size = batch_size
# #         self.current_ind = 0 # 下标
# #         if pad_with_last_sample:
# #             # 下面这些操作就是说：我们的数据在最后一个batch时，不够了，那就用最后一个样本 填满 成一个 batch
# #             num_padding = (batch_size - (len(xs) % batch_size)) % batch_size # 要填充的个数
# #             x_padding = np.repeat(xs[-1:], num_padding, axis=0) # 将最后一个特征样本重复num_padding次：最后一个样本、重复次数、在0维度（最外层）
# #             y_padding = np.repeat(ys[-1:], num_padding, axis=0) # 将最后一个标签样本重复num_padding次：同上
# #             xs = np.concatenate([xs, x_padding], axis=0) # 拼接到原来的特征数据中
# #             ys = np.concatenate([ys, y_padding], axis=0) # 拼接到原来的标签数据中
# #         self.size = len(xs) # 得到 填充 后的 数据集的 大小
# #         self.num_batch = int(self.size // self.batch_size) # 一共需要的batch的个数
# #
# #         # 打乱数据操作
# #         if shuffle:
# #             permutation = np.random.permutation(self.size)
# #             xs, ys = xs[permutation], ys[permutation]
# #         self.xs = xs
# #         self.ys = ys
# #
# #     def get_iterator(self):
# #         """
# #         作用：得到一个迭代器,用于一个batch 一个batch的把数据取出来
# #         """
# #         self.current_ind = 0 # 下标
# #
# #         def _wrapper(): # 是个生成器，节约内存
# #             """
# #             作用：每次取一个batch的数据（之所以没按常规的来写代码，是因为yield可以节约内存）
# #             带有 yield 的函数不再是一个普通函数，Python 解释器会将其视为一个 generator，也就是生成器
# #             理解：yield就是 return 返回一个值，并且记住这个返回的位置，下次迭代就从这个位置后开始。
# #             """
# #             while self.current_ind < self.num_batch: # batch的个数
# #                 start_ind = self.batch_size * self.current_ind # 开始下标
# #                 end_ind = min(self.size, self.batch_size * (self.current_ind + 1)) # 结束下标：self.size为数据集的长度。使用min是因为怕超过了数据集的长度
# #                 x_i = self.xs[start_ind: end_ind, ...] # 切片 取数据
# #                 y_i = self.ys[start_ind: end_ind, ...]
# #                 yield (x_i, y_i) # 返回数据集，并且下次迭代时从这个位置继续
# #                 self.current_ind += 1
# #
# #         return _wrapper() # 这个不是调用，返回的是生成器
#
#
# class StandardScaler:
#     """
#     Standard the input
#     """
#
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std
#
#     def transform(self, data):  # 标准化数据
#         return (data - self.mean) / self.std
#
#     def inverse_transform(self, data): # 恢复数据
#         return (data * self.std) + self.mean
#
#
# def add_simple_summary(writer, names, values, global_step):
#     """
#     Writes summary for a list of scalars.
#     :param writer:
#     :param names:
#     :param values:
#     :param global_step:
#     :return:
#     """
#     for name, value in zip(names, values): # zip将names和values中对应的元素，打包成了元组,返回一个可迭代对象
#         summary = tf.Summary()
#         summary_value = summary.value.add()
#         summary_value.simple_value = value
#         summary_value.tag = name
#         writer.add_summary(summary, global_step)
#
#
# def calculate_normalized_laplacian(adj):
#     """
#     作用：计算正则化的拉普拉斯矩阵矩阵
#     # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
#     # D = diag(A 1)
#     :param adj:
#     :return:
#     """
#     adj = sp.coo_matrix(adj) # 是一种坐标形式的稀疏矩阵。采用三个数组row、col和data保存非零元素的信息，这三个数组的长度相同，row保存元素的行，col保存元素的列，data保存元素的值
#     d = np.array(adj.sum(1)) # 按行求和
#     d_inv_sqrt = np.power(d, -0.5).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#     normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
#     return normalized_laplacian
#
#
# def calculate_random_walk_matrix(adj_mx):
#     """
#     计算随机游走矩阵
#     P = D^-1 * W
#     """
#     # sp.coo_matrix：
#     # 是一种坐标形式的稀疏矩阵。采用三个数组row、col和data保存非零元素的信息，
#     # 这三个数组的长度相同，row保存元素的行，col保存元素的列，data保存元素的值
#     adj_mx = sp.coo_matrix(adj_mx)
#     d = np.array(adj_mx.sum(1))
#     d_inv = np.power(d, -1).flatten() # d的逆
#     d_inv[np.isinf(d_inv)] = 0.
#     d_mat_inv = sp.diags(d_inv)
#     random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
#     return random_walk_mx
#
#
# def calculate_reverse_random_walk_matrix(adj_mx):
#     """
#     作用：计算反向的随机游走矩阵
#     P = D^-1 * W^T
#     """
#     return calculate_random_walk_matrix(np.transpose(adj_mx))
#
#
# def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
#     """
#     作用：将正则化的拉普拉斯矩阵进行压缩
#     """
#     if undirected: # 无向图
#         adj_mx = np.maximum.reduce([adj_mx, adj_mx.T]) # 比较两个矩阵对应元素，谁大就取谁。其实就是把邻接矩阵变成 对称的
#     L = calculate_normalized_laplacian(adj_mx) # 计算正则化的拉普拉斯矩阵
#     if lambda_max is None:
#         # linalg.eigsh()：找出实对称方矩阵或复厄密矩阵A的k个特征值和特征向量。
#         # 下面的L为：对称矩阵、1为：要求的特征值和特征向量的个数、which='LM'表示：求最大特征值
#         # 返回值：k个特征值的数组、表示k个特征向量的数组。
#         lambda_max, _ = linalg.eigsh(L, 1, which='LM')
#         lambda_max = lambda_max[0] # 取列表的第一个特征值
#     L = sp.csr_matrix(L) # 将coo形式 转换为 csr形式
#     M, _ = L.shape
#     I = sp.identity(M, format='csr', dtype=L.dtype) # csr格式，M形状的单位矩阵
#     L = (2 / lambda_max * L) - I
#     return L.astype(np.float32)
#
#
# def config_logging(log_dir, log_filename='info.log', level=logging.INFO):
#     # Add file handler and stdout handler
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     # Create the log directory if necessary.
#     try:
#         os.makedirs(log_dir)
#     except OSError:
#         pass
#     file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
#     file_handler.setFormatter(formatter)
#     file_handler.setLevel(level=level)
#     # Add console handler.
#     console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#     console_handler = logging.StreamHandler(sys.stdout)
#     console_handler.setFormatter(console_formatter)
#     console_handler.setLevel(level=level)
#     logging.basicConfig(handlers=[file_handler, console_handler], level=level)
#
#
# def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
#     """
#     作用：记录日志
#     """
#     logger = logging.getLogger(name) # 记录器名称跟踪包或模块的层次结构，并且直观地从记录器名称显示记录事件的位置。
#     logger.setLevel(level)
#     # Add file handler and stdout handler
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
#     file_handler.setFormatter(formatter)
#     # Add console handler.
#     console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#     console_handler = logging.StreamHandler(sys.stdout)
#     console_handler.setFormatter(console_formatter)
#     logger.addHandler(file_handler)
#     logger.addHandler(console_handler)
#     # Add google cloud log handler
#     logger.info('Log directory: %s', log_dir)
#     return logger
#
#
# def get_total_trainable_parameter_size():
#     """
#     Calculates the total number of trainable parameters in the current graph.计算当前图中可训练参数的总数
#     :return:
#     """
#     total_parameters = 0
#     for variable in tf.trainable_variables():
#         # shape is an array of tf.Dimension
#         total_parameters += np.product([x.value for x in variable.get_shape()])
#     return total_parameters
#
#
# def load_dataset(dataset_dir, batch_size, test_batch_size=None, **kwargs):
#     data1 = np.load('D:/mul-station tide level/GCRNN_PyTorch-main/data/Mexico_province/Mexico_temperature/42002h3.npy')
#
#     scaler = MinMaxScaler(feature_range=(0, 1))
#
#     a = data1.shape[1]
#     d = data1.shape[2]
#     data1 = data1.reshape(-1,d)
#     scaler = scaler.fit(data1)
#
#     data = {}
#     # train_dataset = CustomDataset(dataset_dir, 'train')
#     val_dataset = CustomDataset(dataset_dir, 'val')
#     test_dataset = CustomDataset(dataset_dir, 'test')
#
#
#     # data['train_loader'] = CustomDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     data['val_loader'] = CustomDataLoader(val_dataset, batch_size=test_batch_size, shuffle=False)
#     data['test_loader'] = CustomDataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
#
#
#
#     # 下面调用自己写的DataLoader类
#     # data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle= True)
#     # data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size, shuffle=False)
#     # data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size, shuffle=False)
#     data['scaler'] = scaler
#
#     return data
#
#
# def load_graph_data(pkl_filename):
#     # 分别是传感器节点的实际编号；传感器节点的相对编号（0-206）；邻接矩阵
#     sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
#     return sensor_ids, sensor_id_to_ind, adj_mx
#
#
# def load_pickle(pickle_file):
#     try:
#         with open(pickle_file, 'rb') as f:
#             pickle_data = pickle.load(f)
#     except UnicodeDecodeError as e:
#         with open(pickle_file, 'rb') as f:
#             pickle_data = pickle.load(f, encoding='latin1')
#     except Exception as e:
#         print('Unable to load data ', pickle_file, ':', e)
#         raise
#     return pickle_data  # list类型,包含三个元素
# #

import logging
import numpy as np
import os
import pickle
import scipy.sparse as sp
import sys
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import linalg


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
                                     用最后一个样本填充，使样本数量可被batch_size整除
        """
        self.batch_size = batch_size
        self.current_ind = 0 # 下标
        if pad_with_last_sample:
            # 下面这些操作就是说：我们的数据在最后一个batch时，不够了，那就用最后一个样本 填满 成一个 batch
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size # 要填充的个数
            x_padding = np.repeat(xs[-1:], num_padding, axis=0) # 将最后一个特征样本重复num_padding次：最后一个样本、重复次数、在0维度（最外层）
            y_padding = np.repeat(ys[-1:], num_padding, axis=0) # 将最后一个标签样本重复num_padding次：同上
            xs = np.concatenate([xs, x_padding], axis=0) # 拼接到原来的特征数据中
            ys = np.concatenate([ys, y_padding], axis=0) # 拼接到原来的标签数据中
        self.size = len(xs) # 得到 填充 后的 数据集的 大小
        self.num_batch = int(self.size // self.batch_size) # 一共需要的batch的个数

        # 打乱数据操作
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys = xs[permutation], ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        """
        作用：得到一个迭代器,用于一个batch 一个batch的把数据取出来
        """
        self.current_ind = 0 # 下标

        def _wrapper(): # 是个生成器，节约内存
            """
            作用：每次取一个batch的数据（之所以没按常规的来写代码，是因为yield可以节约内存）
            带有 yield 的函数不再是一个普通函数，Python 解释器会将其视为一个 generator，也就是生成器
            理解：yield就是 return 返回一个值，并且记住这个返回的位置，下次迭代就从这个位置后开始。
            """
            while self.current_ind < self.num_batch: # batch的个数
                start_ind = self.batch_size * self.current_ind # 开始下标
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1)) # 结束下标：self.size为数据集的长度。使用min是因为怕超过了数据集的长度
                x_i = self.xs[start_ind: end_ind, ...] # 切片 取数据
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i) # 返回数据集，并且下次迭代时从这个位置继续
                self.current_ind += 1

        return _wrapper() # 这个不是调用，返回的是生成器


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):  # 标准化数据
        return (data - self.mean) / self.std

    def inverse_transform(self, data): # 恢复数据
        return (data * self.std) + self.mean


def add_simple_summary(writer, names, values, global_step):
    """
    Writes summary for a list of scalars.
    :param writer:
    :param names:
    :param values:
    :param global_step:
    :return:
    """
    for name, value in zip(names, values): # zip将names和values中对应的元素，打包成了元组,返回一个可迭代对象
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        writer.add_summary(summary, global_step)


def calculate_normalized_laplacian(adj):
    """
    作用：计算正则化的拉普拉斯矩阵矩阵
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj) # 是一种坐标形式的稀疏矩阵。采用三个数组row、col和data保存非零元素的信息，这三个数组的长度相同，row保存元素的行，col保存元素的列，data保存元素的值
    d = np.array(adj.sum(1)) # 按行求和
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    """
    计算随机游走矩阵
    P = D^-1 * W
    """
    # sp.coo_matrix：
    # 是一种坐标形式的稀疏矩阵。采用三个数组row、col和data保存非零元素的信息，
    # 这三个数组的长度相同，row保存元素的行，col保存元素的列，data保存元素的值
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten() # d的逆
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    """
    作用：计算反向的随机游走矩阵
    P = D^-1 * W^T
    """
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    """
    作用：将正则化的拉普拉斯矩阵进行压缩
    """
    if undirected: # 无向图
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T]) # 比较两个矩阵对应元素，谁大就取谁。其实就是把邻接矩阵变成 对称的
    L = calculate_normalized_laplacian(adj_mx) # 计算正则化的拉普拉斯矩阵
    if lambda_max is None:
        # linalg.eigsh()：找出实对称方矩阵或复厄密矩阵A的k个特征值和特征向量。
        # 下面的L为：对称矩阵、1为：要求的特征值和特征向量的个数、which='LM'表示：求最大特征值
        # 返回值：k个特征值的数组、表示k个特征向量的数组。
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0] # 取列表的第一个特征值
    L = sp.csr_matrix(L) # 将coo形式 转换为 csr形式
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype) # csr格式，M形状的单位矩阵
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)


def config_logging(log_dir, log_filename='info.log', level=logging.INFO):
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Create the log directory if necessary.
    try:
        os.makedirs(log_dir)
    except OSError:
        pass
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level=level)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level=level)
    logging.basicConfig(handlers=[file_handler, console_handler], level=level)


def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    """
    作用：记录日志
    """
    logger = logging.getLogger(name) # 记录器名称跟踪包或模块的层次结构，并且直观地从记录器名称显示记录事件的位置。
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger


def get_total_trainable_parameter_size():
    """
    Calculates the total number of trainable parameters in the current graph.计算当前图中可训练参数的总数
    :return:
    """
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        total_parameters += np.product([x.value for x in variable.get_shape()])
    return total_parameters


def load_dataset(dataset_dir, batch_size, test_batch_size=None, **kwargs):
    data1 = np.load('D:/mul-station tide level/GCRNN_PyTorch-main/data/Mexico_province/Mexico_temperature/42002h3.npy')
    data1 = data1[:, :, [0,1,4,5]]

    data1 = data1.transpose(1, 0, 2)

    scaler = MinMaxScaler(feature_range=(0, 1))

    a = data1.shape[1]
    d = data1.shape[2]
    data1 = data1.reshape(-1,d)
    scaler = scaler.fit(data1)

    data = {}


    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz')) # 加载数据集
        data['x_' + category] = cat_data['x'] # 特征
        data['y_' + category] = cat_data['y'] # 标签
    # 下面是实例化了一个标准化类的对象
    # scaler = StandardScaler(mean=data['x_train'].mean(), std=data['x_train'].std())
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # a = data['x_train'].shape[0]
    # b= data['x_train'].shape[1]
    # c= data['x_train'].shape[2]
    # d = data['x_train'].shape[3]
    # e = data['y_train'].shape[1]
    # cc = data['x_train'].reshape((-1,d))
    # scaler = scaler.fit(cc)
    #
    # block_size = 100  # 每个分块的大小
    #
    #
    # for category in ['train', 'val', 'test']:
    #
    #     num_blocks = len(data['x_' + category]) // block_size + 1
    #
    #     for block_idx in range(num_blocks):
    #         start_idx = block_idx * block_size
    #         end_idx = min((block_idx + 1) * block_size, len(data['x_' + category]))
    #         y_block = np.concatenate(
    #             (data['x_' + category][start_idx:end_idx][:, :, :, :(data['x_' + category].shape[3] - 1)], data['y_' + category][start_idx:end_idx],), axis=3)
    #
    #
    #
    #         x_block = scaler.transform(data['x_' + category][start_idx:end_idx].reshape((-1, d)))  # 特征标准化
    #         y_block = scaler.transform(y_block.reshape((-1, d)))  # 标签标准化
    #
    #         y_block = y_block[..., (d - 1):]  # 标签截取
    #
    #         x_block = x_block.reshape((-1, b, c, d))  # 调整特征数据形状
    #         y_block = y_block.reshape((-1, e, c, 1))  # 调整标签数据形状
    #
    #         data['x_' + category][start_idx:end_idx] = x_block
    #         data['y_' + category][start_idx:end_idx] = y_block

    # # Data format
    # for category in ['train', 'val', 'test']:
    #     data['x_' + category] = scaler.transform(data['x_' + category].reshape((-1,d))) # 特征标准化
    #     data['y_' + category] = scaler.transform(data['y_' + category].reshape((-1,d))) # 标签标准化
    # for category in ['train', 'val', 'test']:
    #
    #     data['y_' + category] = data['y_' + category][...,(d-1):] # 标签
    #
    # for category in ['train', 'val', 'test']:
    #     data['x_' + category] = data['x_' + category].reshape((-1, b, c, d))
    #
    #     data['y_' + category] = data['y_' + category].reshape((-1,e,c,1))  # 标签



    # 下面调用自己写的DataLoader类
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle= True)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size, shuffle=False)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size, shuffle=False)
    data['scaler'] = scaler

    return data


def load_graph_data(pkl_filename):
    # 分别是传感器节点的实际编号；传感器节点的相对编号（0-206）；邻接矩阵
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data  # list类型,包含三个元素
