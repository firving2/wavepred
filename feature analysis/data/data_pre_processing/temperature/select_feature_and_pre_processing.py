from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd
import os

def save(output_filename, data):
    """保存文件"""
    np.save(output_filename, data)
    print("save file success!")

def padding_abnormal_data(df, output_filename):
    # 将异常值999999替换为nan
    df = df.replace(to_replace=999999, value=np.nan)

    # 下面是检查是否有缺失值
    # print(pd.notnull(df))
    print("没填充之前------------------------")
    print(np.all(pd.notnull(df)))  # 返回False说明有

    # 这个循环，每次取出一列数据，然后用均值来填充
    for i in df.columns:
        if np.all(pd.notnull(df[i])) == False:
            df[i].fillna(df[i].mean(), inplace=True)

    # 再次验证
    print("填充之后---------------------")
    print(np.all(pd.notnull(df)))  # 返回True说明没有缺失值

    # 转换为numpy类型，并且划分维度
    data = df.values  # numpy.ndarray
    print(data.shape)
    data = data.reshape(7, -1, 6)
    print(data[0,0:10,:])
    print(data.shape)

    # 保存
    save(output_filename, data)


def select_feature(filename, output_filename):
    df = pd.read_csv(filename)
    print(df.values.shape)
    # Details please see data/SURF_CHN_MUL_HOR_readme.txt
    # 气温、最高气温、最低气温、体感温度、气压、最大风速、水汽压、降水量、水平能见度(人工)、现在天气、风力(11个特征)
    df = df.loc[:, ['u10','v10','mwp','sst_data','sp_data','swh_data']]#u10,v10,mwp,sst_data,sp_data,swh_data

    padding_abnormal_data(df, output_filename)

def main(temperature_df_filename, output_filename):
    print("selecting feature-----------------")
    select_feature(temperature_df_filename, output_filename)

if __name__ == "__main__":
    temperature_df_filename = r"D:\mul-station tide level\why factors\GCRNN_PyTorch-main\data\Mexico_province\Mexico_temperature\42002h3.csv"
    output_filename = r"D:\mul-station tide level\why factors\GCRNN_PyTorch-main\data\Mexico_province\Mexico_temperature\42002h3"
    main(temperature_df_filename, output_filename)

    print("finish!")