import logging
import numpy as np
import os
import pickle
import scipy.sparse as sp
import sys
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import linalg
import math
import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, dataset_dir, category):
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        self.features = torch.tensor(cat_data['x'], dtype=torch.float32)
        new_data = {}
        new_data['y'] = cat_data['y']
        new_data['y'] = new_data['y'][..., -1:]
        self.labels = torch.tensor(new_data['y'], dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        self.features = self.features.permute(1, 0, 2, 3)
        self.labels = self.labels.permute(1, 0, 2, 3)
        return self.features[idx], self.labels[idx]


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False,
                 drop_last=False, timeout=0, worker_init_fn=None):
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers,
                         collate_fn, pin_memory, drop_last, timeout, worker_init_fn)

        self.size = len(dataset)
        self.num_batch = math.ceil(self.size / self.batch_size)


def load_dataset(dataset_dir, batch_size, test_batch_size=None, **kwargs):
    data1 = np.load('D:/mul-station tide level/GCRNN_PyTorch-main/data/Mexico_province/Mexico_temperature/42002h3.npy')

    scaler = MinMaxScaler(feature_range=(0, 1))

    a = data1.shape[1]
    d = data1.shape[2]
    data1 = data1.reshape(-1,d)
    scaler = scaler.fit(data1)

    data = {}
    train_dataset = CustomDataset(dataset_dir, 'train')
    val_dataset = CustomDataset(dataset_dir, 'val')
    test_dataset = CustomDataset(dataset_dir, 'test')

    data['train_loader'] = CustomDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    data['val_loader'] = CustomDataLoader(val_dataset, batch_size=test_batch_size, shuffle=False)
    data['test_loader'] = CustomDataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)



    # 下面调用自己写的DataLoader类
    # data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle= True)
    # data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size, shuffle=False)
    # data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size, shuffle=False)
    data['scaler'] = scaler

    return data



