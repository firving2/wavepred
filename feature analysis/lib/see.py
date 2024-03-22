import logging
import numpy as np
import os
import pickle
import scipy.sparse as sp
import sys
import tensorflow as tf
cat_data = np.load(os.path.join('D:\mul-station tide level\GCRNN_PyTorch-main\data\Heilongjiang_province\Heilongjiang_temperature',
                                'train'+'.npz'))
print(cat_data.files)
print(cat_data['y'].shape)