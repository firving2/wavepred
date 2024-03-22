from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from scipy import interpolate
import scienceplots
import argparse
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import os
from scipy.stats import pearsonr
def stack_data_in_batches(data_list, batch_size):
    stacked_data = []
    num_batches = len(data_list) // batch_size

    for i in range(num_batches):
        batch_data = data_list[i * batch_size: (i + 1) * batch_size]
        batch_data = np.stack(batch_data, axis=0)
        stacked_data.append(batch_data)

    remaining_data = data_list[num_batches * batch_size:]
    if remaining_data:
        remaining_data = np.stack(remaining_data, axis=0)
        stacked_data.append(remaining_data)

    return np.concatenate(stacked_data, axis=0)




def generate_graph_seq2seq_io_data(temperature_data, x_offsets, y_offsets, scaler=None):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """
    data = temperature_data
    print(data.shape)
    num_samples, num_nodes, features = data.shape

    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets))) # Exclusive

    # x_shape = (max_t - min_t, 12,666,7)  # 根据实际情况定义 x 的形状
    # y_shape = (max_t - min_t, 12,666,1)  # 根据实际情况定义 y 的形状
    #
    # x = np.empty(x_shape, dtype=data.dtype)
    # y = np.empty(y_shape, dtype=data.dtype)
    # for t in range(min_t, max_t):
    #     x_t = data[t + x_offsets, ...]
    #     u_t = data[t+y_offsets,:,0:1]
    #     v_t = data[t+y_offsets,:,1:2]
    #     x_t = np.concatenate([x_t,u_t,v_t],axis=2)
    #     y_t = data[t + y_offsets, :,-1:]
    #     #
    #     # x[t - min_t] = x_t
    #     # y[t - min_t] = y_t
    #
    # # for t in range(min_t, max_t):
    # #     x_t = data[t + x_offsets, ...]
    # #     y_t = data[t + y_offsets, ...]
    # #
    #     x.append(x_t)
    #     y.append(y_t)
    # #
    # #     # x_t = data[in_start + x_offsets, ...]
    # #     # y_t = data[in_start + y_offsets, ...]
    # #     # in_start += 72
    # #     # x.append(x_t)
    # #     # y.append(y_t)
    # #
    # x = np.stack(x, axis=0)
    # y = np.stack(y, axis=0)

    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, :,-1:]
        #
        # x[t - min_t] = x_t
        # y[t - min_t] = y_t

    # for t in range(min_t, max_t):
    #     x_t = data[t + x_offsets, ...]
    #     y_t = data[t + y_offsets, ...]
    #
        x.append(x_t)
        y.append(y_t)
    #
    #     # x_t = data[in_start + x_offsets, ...]
    #     # y_t = data[in_start + y_offsets, ...]
    #     # in_start += 72
    #     # x.append(x_t)
    #     # y.append(y_t)
    #
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y

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
            y = np.mean(data[in_start:in_end,:,-1:],axis=0)
            y = np.mean(y,axis=0)
            # mean = mean.reshape(1,mean.shape[0])
            Y.append(y)
        in_start = in_start+split
    X = np.stack(X, axis=0)
    Y = np.stack(Y, axis=0)

    return X,Y







def generate_train_val_test(args):
    """
    generate train data,val data and test data
    :param args:
    :return:
    """
    data = np.load(args.temperature_filename)

    print(data.shape)
    data = data.transpose(1, 0, 2)


    #
    # scaler = MinMaxScaler(feature_range=(0, 1))
    #
    # a = data.shape[1]
    # d = data.shape[2]
    # data = data.reshape(-1,d)
    # scaler = scaler.fit(data)
    #
    # data = scaler.transform(data)
    # print(data)
    # data = data.reshape(-1,a,d)

    # 0 is the latest observed sample.
    x_offsets =np.sort(
        np.concatenate((np.arange(-5, 1, 1),))
    )
    # print(x_offsets) # [-11 -10  -9  -8  -7  -6  -5  -4  -3  -2  -1   0]
    # Predict the next 12 hour.
    y_offsets = np.sort(np.arange(1, 7, 1))
    # print(y_offsets) # [ 1  2  3  4  5  6  7  8  9 10 11 12]

    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y  = supervised_data(data,24,24,24)

    # x, y = generate_graph_seq2seq_io_data(data, x_offsets=x_offsets, y_offsets=y_offsets,)

    # x = stack_data_in_batches(x[:10000], 100)
    # y = np.stack(y[:10000], axis=0)
    # x = np.array(x)
    # y = np.array(y)

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    # 7/10 is used for training, 1/10 is used for validation and 2/10 is used for testing
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train =round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train =x[:num_train], y[:num_train]

    # val
    x_val, y_val =(x[num_train:num_train+num_val], y[num_train:num_train+num_val])
    # test
    x_test, y_test =x[-num_test:], y[-num_test:]

    n_samples = x_train.shape[0]
    n_features = np.prod(x_train.shape[1:])
    input_data_reshaped = x_train.reshape(n_samples, n_features)

    # 构建线性回归模型
    model = LinearRegression()

    # 拟合模型
    model.fit(input_data_reshaped, y_train)

    x_test = x_test[:]
    y_test = y_test[:]

    n_samples = x_test.shape[0]
    n_features = np.prod(x_test.shape[1:])
    input_data_reshaped = x_test.reshape(n_samples, n_features)

    # 预测目标变量
    inv_yhat = model.predict(input_data_reshaped)
    inv_yhat = inv_yhat.reshape(-1,)
    inv_y  = y_test.reshape(-1,)

    year5_true = inv_y[-365:]
    year5_predicted = inv_yhat[-365:]

    year4_true = inv_y[-365 * 2:-365]
    year4_predicted = inv_yhat[-365 * 2:-365]

    year3_true = inv_y[-(365 * 2 + 366):-365 * 2]
    year3_predicted = inv_yhat[-(365 * 2 + 366):-365 * 2]

    year2_true = inv_y[-(365 * 2 + 366 + 365):-(365 * 2 + 366)]
    year2_predicted = inv_yhat[-(365 * 2 + 366 + 365):-(365 * 2 + 366)]

    year1_true = inv_y[-(365 * 2 + 366 + 365 + 365):-(365 * 2 + 366 + 365)]
    year1_predicted = inv_yhat[-(365 * 2 + 366 + 365 + 365):-(365 * 2 + 366 + 365)]

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



    #
    # matplotlib.rcParams.update({'font.size': 42})
    # with plt.style.context(['science', 'no-latex', 'cjk-sc-font']):
    #     fig, ax = plt.subplots()
    #     fig.set_size_inches(10, 8)  # 设置图像的长宽比
    #     ax.plot(y_test, label='Target',linewidth=2)
    #     ax.plot(y_test_hat, label='LinearRegression',linewidth=2)
    #     ax.legend(loc="upper right")
    #     ax.set_xlabel('Time (day)')
    #     ax.set_ylabel('Daily average regional \nwave height (m)')
    #     ax.autoscale(tight=True)
    #     ax.set_ylim(top=3)
    #     for spine in ax.spines.values():
    #         spine.set_linewidth(2)  # Adjust the linewidth as desired
    #     ax.tick_params(axis='both', which='both', width=2)
    #
    #     fig.savefig('LinearRegression.tif', dpi=300)
    #
    # plt.show()










    # for cat in ["train", "val", "test"]:
    #     _x, _y = locals()["x_" + cat], locals()["y_" + cat]
    #     print(cat, "x: ", _x.shape, "y: ", _y.shape)
    #     np.savez_compressed(
    #         os.path.join(args.output_dir, "%s.npz" % cat),
    #         x=_x,
    #         y=_y,
    #         # x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
    #         # y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
    #     )


def main(args):
    print("Generating training data")
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="D:/mul-station tide level/why factors/GCRNN_PyTorch-mainmul-var/data/Mexico_province/Mexico_temperature/", help="Output directory.")
    parser.add_argument("--temperature_filename", type=str,
                        default="D:/mul-station tide level/why factors/GCRNN_PyTorch-mainmul-var/data/Mexico_province/Mexico_temperature/42002h3.npy",
                        help="Raw temperature readings."
    )
    args = parser.parse_args() # Get all parameters
    main(args)