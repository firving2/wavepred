import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt

import pandas as pd
def visualization_dcrnn_prediction(filename: str):


    f = np.load(filename)


    prediction = f["prediction"][:,:5251,:] # (12, 256, 74)

    data2 = np.load('D:\mul-station tide level\GCRNN_PyTorch-main\data\Mexico_province\Mexico_temperature/test.npz')
    data2 = data2['y']
    print(data2.shape)


    # print(prediction[:,4,1])
    # print(prediction[:, 5, 1])
    truth = f["truth"][:,:5251,:] # (12, 256, 74)
    print(truth.shape)

    inv_y = truth[:, :,:]

    inv_yhat = prediction[:, :, :]


    inv_y1 = inv_y.transpose(1, 0,2)
    print(inv_y.shape)

    inv_yhat1 = inv_yhat.transpose(1,0,2)

    inv_y = inv_y1[::12,:, 100]
    inv_yhat = inv_yhat1[::12,:,100]
    # # print(inv_y.shape)
    # inv_y = inv_y.reshape(-1,1)
    # # # # inv_y = inv_y[:,:]
    # # # # print(inv_y.shape)
    inv_y = inv_y.reshape(-1,)
    # # # # # inv_y = np.nan_to_num(inv_y, nan=0)
    # # # #
    # # # #
    # # # #
    # # # #
    # # inv_yhat = inv_yhat1.reshape(-1,inv_yhat1.shape[2])
    # # # # inv_yhat = inv_yhat[:,:]
    inv_yhat = inv_yhat.reshape(-1,)
    # # inv_yhat = np.nan_to_num(inv_yhat, nan=0)
    #
    #
    # if np.isnan(inv_yhat).any():
    #     print("数组包含 NaN")
    # else:
    #     print("数组不包含 NaN")
    # # #
    # S = 0
    # MAE = 0
    # M = inv_y.shape[0]
    # S1 = 0
    # CC=0
    # for i in range(inv_y.shape[0]):
    #     # print(inv_y.shape)
    #     inv_y1 = inv_y[i, :]
    #     inv_yhat1 = inv_yhat[i, :]
    #     inv_y1 = inv_y1.reshape(-1, )
    #     inv_yhat1 = inv_yhat1.reshape(-1, )
    #
    #     s = 0
    #     mae = 0
    #     m = inv_y1.shape[0]
    #     cc = 0
    #
    #     for j in range(inv_y1.shape[0]):
    #         mae += abs(inv_y1[j] - inv_yhat1[j])
    #         s += (inv_y1[j] - inv_yhat1[j]) ** 2
    #     mae = mae / m
    #     s1 = s / m
    #     s = sqrt(s1)
    #     # X1 = pd.Series(inv_y1)
    #     # Y1 = pd.Series(inv_yhat1)
    #     # if X1.var() == 0 or Y1.var() == 0:
    #     #     cc = 0  # 或者使用其他默认值
    #     # else:
    #     #     cc = X1.corr(Y1, method="pearson")
    #     # cc = X1.corr(Y1, method="pearson")
    #     print(cc)
    #
    #     MAE += mae
    #     S  += s
    #
    #     S1  += s1
    #     CC += cc
    #
    # S = S/M
    # S1 = S1/M
    # MAE = MAE/M
    # CC = CC/M
    #
    # print(round(CC, 4))
    # print(round(S, 4))
    # print(round(S1, 4))
    # print(round(MAE, 4))


    #
    # S = 0
    # MAE = 0
    # M = inv_y.shape[0]
    # S1 = 0
    # CC=0
    # for i in range(inv_y.shape[0]):
    #     # print(inv_y.shape)
    #     inv_y1 = inv_y[i,:, :]
    #     inv_yhat1 = inv_yhat[i,:, :]
    #     # inv_y1 = inv_y1.reshape(-1, )
    #     # inv_yhat1 = inv_yhat1.reshape(-1, )
    #
    #     s = 0
    #     mae = 0
    #     m = inv_y1.shape[0]*inv_y1.shape[1]
    #     print(m)
    #     cc = 0
    #
    #     for j in range(inv_y1.shape[0]):
    #         inv_y2 = inv_y1[j]
    #         inv_yhat2 = inv_yhat1[j]
    #         for c in range(inv_y1.shape[1]):
    #             mae += abs(inv_y2[c] - inv_yhat2[c])
    #             s += (inv_y2[c] - inv_yhat2[c]) ** 2
    #             # X1 = pd.Series(inv_y2)
    #             # Y1 = pd.Series(inv_yhat2)
    #             # if X1.var() == 0 or Y1.var() == 0:
    #             #     cc = 0  # 或者使用其他默认值
    #             # else:
    #             #     cc = X1.corr(Y1, method="pearson")
    #     mae = mae / m
    #     s1 = s / m
    #     s2 = sqrt(s1)
    #
    #
    #     #     mae += abs(inv_y1[j] - inv_yhat1[j])
    #     #     s += (inv_y1[j] - inv_yhat1[j]) ** 2
    #     # mae = mae / m
    #     # s1 = s / m
    #     # s = sqrt(s1)
    #     # X1 = pd.Series(inv_y1)
    #     # Y1 = pd.Series(inv_yhat1)
    #     # if X1.var() == 0 or Y1.var() == 0:
    #     #     cc = 0  # 或者使用其他默认值
    #     # else:
    #     #     cc = X1.corr(Y1, method="pearson")
    #     # # cc = X1.corr(Y1, method="pearson")
    #
    #
    #     MAE += mae
    #     S  += s2
    #
    #     S1  += s1
    #     # CC += cc
    #
    # S = S/M
    # S1 = S1/M
    # MAE = MAE/M
    # # CC = CC/M
    #
    # print(round(CC, 4))
    # print(round(S, 4))
    # print(round(S1, 4))
    # print(round(MAE, 4))


    #
    # #
    # #
    # s = 0
    # mae = 0
    # m = inv_y.shape[0]
    # #
    # for i in range(inv_y.shape[0]):
    #     mae += abs(inv_y[i] - inv_yhat[i])
    #     s += (inv_y[i] - inv_yhat[i])**2
    # mae = mae / m
    # s1 = s / m
    # s2 = sqrt(s1)
    # X1 = pd.Series(inv_y)
    # Y1 = pd.Series(inv_yhat)
    # cc = X1.corr(Y1, method="pearson")
    # print(round(cc, 4))
    # print(round(s1, 4))
    # print(round(s2, 4))
    # print(round(mae, 4))
    #
    #
    #
    plt.plot(inv_y)
    plt.plot(inv_yhat)
    #
    #
    #
    plt.legend(["target", "prediction"], loc="upper right")

    plt.show()


if __name__ == "__main__":
    visualization_dcrnn_prediction("../data/fcrnn_heilongjiang_predictions.npz")