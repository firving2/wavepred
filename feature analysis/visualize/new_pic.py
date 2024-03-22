import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import linregress
import pandas as pd
import matplotlib
from scipy import interpolate
import scienceplots

def supervised_data(temperature_data,n_input,n_output,split):
    data = temperature_data

    X,Y = list(),list()
    in_start = 0
    for _ in range(len(data)):
        in_end = in_start+n_input
        out_end = in_start+n_output
        if out_end<=len(data):
            x_input = data[in_start:in_end,:,:-1]
            X.append(x_input)
            mean = np.mean(data[in_start:in_end,:,-1:],axis=0)
            mean = np.mean(mean,axis=0)
            # mean = mean.reshape(1,mean.shape[0])
            Y.append(mean)
        in_start = in_start+split
    X = np.stack(X, axis=0)
    Y = np.stack(Y, axis=0)

    return Y

def visualization_dcrnn_prediction(filename: str):


    f = np.load(filename)
    print(f["prediction"][-1])

    prediction = f["prediction"][:,:] # (12, 256, 74)

    # data1 = np.load( 'D:/mul-station tide level/why factors/GCRNN_PyTorch-main/data/Mexico_province/Mexico_temperature/42002h3.npy')
    # data1 = data1.transpose(1, 0, 2)
    # swh = supervised_data(data1,24,24,24)
    #
    # scaler = MinMaxScaler(feature_range=(0, 1))
    #
    # a = data1.shape[1]
    # d = data1.shape[2]
    # data1 = data1.reshape(-1,d)
    # scaler = scaler.fit(data1)
    # # data1 = scaler.transform(data1)
    # mm = np.concatenate([data1[:swh.shape[0],:-1],swh],axis=1)
    #
    # # mm = scaler.transform(mm)
    # swh = mm[-2118:,-1:]


    # data2 = np.load('D:\mul-station tide level\why factors\GCRNN_PyTorch-mainmul-var\data\Mexico_province\Mexico_temperature/42002h3.npy')
    # # data2 = data2['x']
    # print(data2[1,-1,0])


    # print(prediction[:,4,1])
    # print(prediction[:, 5, 1])
    truth = f["truth"][:,:] # (12, 256, 74)
    print(truth)

    inv_y1 = truth[:, :]

    inv_yhat1 = prediction[:,  :]


    # inv_y1 = inv_y.transpose(1, 0,2)
    # print(inv_y1.shape)
    #
    # inv_yhat1 = inv_yhat.transpose(1,0,2)

    # inv_y = inv_y1[:,:, :]
    # inv_yhat = inv_yhat1[:,:,:]
    # # print(inv_y.shape)

    # # # # inv_y = inv_y[:,:]
    # # # # print(inv_y.shape)
    inv_y = inv_y1.reshape(-1,)
    # # # # # inv_y = np.nan_to_num(inv_y, nan=0)
    # # # #
    # # # #
    # # # #
    # # # #

    # # # # inv_yhat = inv_yhat[:,:]
    inv_yhat = inv_yhat1.reshape(-1,)
    # 假设有五年的真实值和预测值数据，每年的数据存储在一个列表中
    year5_true = inv_y[-365:]
    year5_predicted = inv_yhat[-365:]

    year4_true = inv_y[-365*2:-365]
    year4_predicted = inv_yhat[-365*2:-365]

    year3_true = inv_y[-(365*2+366):-365*2]
    year3_predicted = inv_yhat[-(365*2+366):-365*2]

    year2_true = inv_y[-(365*2+366+365):-(365*2+366)]
    year2_predicted = inv_yhat[-(365*2+366+365):-(365*2+366)]

    year1_true = inv_y[-(365*2+366+365+365):-(365*2+366+365)]
    year1_predicted = inv_yhat[-(365*2+366+365+365):-(365*2+366+365)]

    # 存储每一年的皮尔逊相关系数
    correlation_coefficients = []

    # 计算每一年的皮尔逊相关系数
    for year_true, year_predicted in [(year1_true, year1_predicted), (year2_true, year2_predicted),
                                      (year3_true, year3_predicted), (year4_true, year4_predicted),
                                      (year5_true, year5_predicted)]:
        correlation_coefficient, _ = pearsonr(year_true, year_predicted)
        correlation_coefficients.append(correlation_coefficient)

    # 打印每一年的皮尔逊相关系数
    for year, correlation_coefficient in enumerate(correlation_coefficients, start=1):
        print(f"Year {year} 皮尔逊相关系数:", correlation_coefficient)
    # 绘制柱状图
    years = ['2014 year', '2015 year', '2016 year', '2017 year', '2018 year']
    plt.bar(years, correlation_coefficients)
    plt.xlabel('Year')
    plt.ylabel('Pearson Correlation Coefficient')
    plt.title('Correlation Coefficients for Each Year')
    plt.show()
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


        #     mae += abs(inv_y1[j] - inv_yhat1[j])
        #     s += (inv_y1[j] - inv_yhat1[j]) ** 2
        # mae = mae / m
        # s1 = s / m
        # s = sqrt(s1)
        # X1 = pd.Series(inv_y1)
        # Y1 = pd.Series(inv_yhat1)
        # if X1.var() == 0 or Y1.var() == 0:
        #     cc = 0  # 或者使用其他默认值
        # else:
        #     cc = X1.corr(Y1, method="pearson")
        # # cc = X1.corr(Y1, method="pearson")


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



    # plt.plot(inv_y)
    # plt.plot(inv_yhat)

    # a = inv_y[900:1018]
    # time_series = pd.date_range(start='2015-07-18', periods=len(a), freq='H')
    # x = np.arange(len(a))
    #
    # # 示例数据
    # timestamps = pd.date_range(start='2015-09-01', periods=len(a), freq='D')  # 生成每隔一个小时的时间戳
    # occlusion_sensitivity = a
    #
    # # 创建 DataFrame，将时间戳和 occlusion sensitivity 值作为列
    # df = pd.DataFrame({'timestamp': timestamps, 'occlusion_sensitivity': occlusion_sensitivity})
    #
    # # 将时间戳作为索引
    # df.set_index('timestamp', inplace=True)
    #
    # # 使用 resample 方法将数据按每十小时聚合，并计算每个聚合区间的平均值
    # resampled_df = df.resample('10D').mean()
    # print(resampled_df)
    #
    # # 创建自变量数组 x，范围为 0 到聚合后的数据长度减 1
    # x = np.arange(len(resampled_df))
    #
    # # 计算线性趋势
    # slope, intercept, r_value, p_value, std_err = linregress(x,resampled_df['occlusion_sensitivity'])
    #
    # # 输出结果
    # print("斜率（线性趋势）：", slope)
    # print("截距：", intercept)
    # print("p值：", p_value)
    #
    #
    #
    # print(a.shape)


    # plt.plot(data2)
    # plt.plot(swh)
    # df = pd.DataFrame(inv_y, columns=['swh'])
    # filename = 'swh.csv'
    # df.to_csv(filename, index=False)
    # slope, intercept, r_value, p_value, std_err = linregress(x,a)
    # print("斜率（线性趋势）：", slope)
    # print("截距：", intercept)
    # print("相关系数：", r_value)
    # print("p值：", p_value)
    # print("标准误差：", std_err)




    # plt.legend(["target", "prediction"], loc="upper right")
    #
    # plt.show()
    # matplotlib.rcParams.update({'font.size': 35})
    # with plt.style.context(['science', 'no-latex', 'cjk-sc-font']):
    #     fig, ax = plt.subplots()
    #     ax.plot(inv_y, label='Target')
    #     ax.plot(inv_yhat, label='GC-GRU model')
    #     ax.legend(loc="upper right")
    #     ax.set_xlabel('Time (day)')
    #     ax.set_ylabel('Daily average regional wave height (m)')
    #     ax.autoscale(tight=True)
    #     ax.set_ylim(top=3)
    #     for spine in ax.spines.values():
    #         spine.set_linewidth(1.5)  # Adjust the linewidth as desired
    #     ax.tick_params(axis='both', which='both', width=1.5)
    #
    # plt.show()
    # matplotlib.rcParams.update({'font.size': 42})
    # with plt.style.context(['science', 'no-latex', 'cjk-sc-font']):
    #     fig, ax = plt.subplots()
    #     fig.set_size_inches(10, 8)  # 设置图像的长宽比
    #     ax.plot(inv_y, label='Target',linewidth=2)
    #     ax.plot(inv_yhat, label='GC-GRU model',linewidth=2)
    #     ax.legend(loc="upper right")
    #     ax.set_xlabel('Time (day)')
    #     ax.set_ylabel('Daily average regional \nwave height (m)')
    #     ax.autoscale(tight=True)
    #     ax.set_ylim(top=3)
    #     for spine in ax.spines.values():
    #         spine.set_linewidth(2)  # Adjust the linewidth as desired
    #     ax.tick_params(axis='both', which='both', width=2)

        # fig.savefig('LinearRegression.tif', dpi=300)

    plt.show()



if __name__ == "__main__":
    visualization_dcrnn_prediction("../data/fcrnn_heilongjiang_predictions.npz")