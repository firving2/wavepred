import argparse
import numpy as np
import os
import sys
import yaml
import pandas as pd
import xarray as xr

from lib import utils

from lib.utils import load_graph_data
from model.gcrnn_supervisor import GCRNNSupervisor
import time
from model.gcrnn_model import GCRNNModel
import torch
import shap
import torch

cuda_device_count = torch.cuda.device_count()
print("CUDA设备数量：", cuda_device_count)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device  = 'cpu'



def prepare_data( x, y):
    """
    作用：准备数据（数据维度的转变，以及放到GPU上等操作）
    """
    x, y = get_x_y(x, y)
    x, y = get_x_y_in_correct_dims(x, y)
    return x.to(device), y.to('cpu')  # 将数据放到GPU上，并返回


def get_x_y( x, y):
    """
    作用：将数据转换为tensor,并转换成对应的维度
    :param x: shape (batch_size, seq_len, num_sensor, input_dim)
    :param y: shape (batch_size, horizon, num_sensor, input_dim)
    :returns x shape (seq_len, batch_size, num_sensor, input_dim)
             y shape (horizon, batch_size, num_sensor, input_dim)
    """
    x = torch.from_numpy(x).float()  # 转换成tensor

    y = torch.from_numpy(y).float()

    # self._logger.debug("X: {}".format(x.size()))  # 记录日志：x的维度
    # self._logger.debug("y: {}".format(y.size()))  # 记录日志：y的维度
    # x = x.permute(1, 0, 2, 3)
    # y = y.permute(1, 0, 2, 3)
    return x, y


def get_x_y_in_correct_dims(x, y):
    """
    作用：调整x,y的维度
    :param x: shape (seq_len, batch_size, num_sensor, input_dim)
    :param y: shape (horizon, batch_size, num_sensor, input_dim)
    :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
             y: shape (horizon, batch_size, num_sensor * output_dim)
    """
    batch_size = x.size(0)
    x = x.view(batch_size,12, 666 * 4)

    # # 将y中最里面的维度进行切片取前output，之后再将最后两维合并成一维（这两维为：节点个数和输出特征）
    # y = y.view(self.horizon, batch_size,
    #                                   self.num_nodes * self.output_dim)

    return x, y

def setup_graph(gcrnn_model,data):
    """
    作用：？？？？？？
    """
    with torch.no_grad(): # 不记录梯度
        gcrnn_model = gcrnn_model.eval().to(device) # 打开测试模式

        # val_iterator = self._data['val_loader'] # 加载验证数据集
        val_iterator = data['val_loader'].get_iterator()

        for _, (x, y) in enumerate(val_iterator):
            x, y = prepare_data(x, y) # 将数据 转换成 符合dcrnn模型的数据
            output = gcrnn_model(x) # 将数据送入模型，得出预测值
            break





def get_log_dir(kwargs):  # 前导单下划线表示：不要在类外访问（类似就私有方法，但只是给程序员的警告，实际能访问）
    """
    作用：创建日志文件夹
    """
    log_dir = kwargs['train'].get('log_dir')
    if log_dir is None:
        batch_size = kwargs['data'].get('batch_size')
        learning_rate = kwargs['train'].get('base_lr')
        max_diffusion_step = kwargs['model'].get('max_diffusion_step')  # 最大的扩散步数
        num_rnn_layers = kwargs['model'].get('num_rnn_layers')  # rnn的层数
        rnn_units = kwargs['model'].get('rnn_units')  # 隐层神经元的个数
        structure = '-'.join(
            ['%d' % rnn_units for _ in range(num_rnn_layers)])
        horizon = kwargs['model'].get('horizon')
        filter_type = kwargs['model'].get('filter_type')  # 滤波器的类型
        filter_type_abbr = 'L'  # 滤波器类型的缩写
        if filter_type == 'random_walk':  # 随机游走
            filter_type_abbr = 'R'
        elif filter_type == 'dual_random_walk':  # 双向的随机游走
            filter_type_abbr = 'DR'
        run_id = 'gcrnn_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/' % (
            filter_type_abbr, max_diffusion_step, horizon,
            structure, learning_rate, batch_size,
            time.strftime('%m%d%H%M%S'))
        base_dir = kwargs.get('base_dir')
        log_dir = os.path.join(base_dir, run_id)  # 拼接成日志的路径
    if not os.path.exists(log_dir):  # 若不存在，则根据前面拼接的路径创建
        os.makedirs(log_dir)
    return log_dir

def shap1(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.safe_load(f)
        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        model_kwargs = supervisor_config.get('model')
        data_lar = supervisor_config.get('data')
        log_dir = get_log_dir(supervisor_config)

        station_ids, station_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)
        log_level = supervisor_config.get('log_level', 'INFO')
        logger = utils.get_logger(log_dir, __name__, 'info.log', level=log_level)
        data = utils.load_dataset(**data_lar)

        gcrnn_model = GCRNNModel(adj_mx, logger, **model_kwargs)
        # gcrnn_model = gcrnn_model.to(device)
        setup_graph(gcrnn_model,data)
        checkpoint = torch.load('models12z/epo217.tar' , map_location='cpu')  # 加载模型参数
        gcrnn_model.load_state_dict(checkpoint['model_state_dict'])  # 将模型参数 加载到模型中


        gcrnn_model = gcrnn_model.eval()
        x = data['x_train']
        y = data['y_train']
        x_test = data['x_test']
        y_test = data['y_test']
        # x = x[:500]
        # y= y[:500]
        subset_size = 500
        subset_indices = np.random.choice(x.shape[0], subset_size, replace=False)
        x = x[subset_indices]
        y = y[subset_indices]
        print(y_test.shape)

        x_test = x_test[200:300]
        y_test = y_test[200:300]
        print(y_test.shape)
        x_test,y_test = prepare_data(x_test,y_test)


        x, y = prepare_data(x, y)
        print(x.shape)


        gcrnn_model = gcrnn_model
        # x = x.to(device)
        feature_names = ['U10 (m/s)', 'V10 (m/s)', 'T2m (°C)', 'Sp (mb)', 'mwp (s)',
                         'Mwd (degree)']
        # y = gcrnn_model(x)


        explainer = shap.GradientExplainer(gcrnn_model, x)
        shap_values = explainer.shap_values(x_test)
        x_test = x_test.cpu().numpy()
        shap_values = np.array(shap_values)
        print(shap_values.shape)
        # shap_values = np.mean(shap_values, axis=1)
        # shap_values =shap_values[-1]#(第6小时） （batch_size, sequence, node, feature)
        # shap_values =shap_values[0]#(第1小时） （batch_size, sequence, node, feature)
        # shap_values = shap_values[1]  # (第2小时） （batch_size, sequence, node, feature)
        shap_values = shap_values.reshape(shap_values.shape[0]*shap_values.shape[1],666,4)


        # np.save('shap_values', shap_values)
        #
        # if 1:  # 对每个结果进行归一化
        #     row_max = shap_values.max(axis=1, keepdims=True)
        #     # 对每行数据进行标准化（除以最大值）
        #     shap_values = shap_values / row_max
        #
        # mean_abs_shap = np.zeros(shap_values[0].shape[1:])
        #
        # # 遍历 SHAP 值
        # for shap_arr in shap_values:
        #     # 取绝对值
        #     abs_shap_arr = np.abs(shap_arr)
        #     # 计算平均绝对值并累加到总和
        #     mean_abs_shap += np.mean(abs_shap_arr, axis=0)
        #
        # # 计算平均绝对值
        # mean_abs_shap /= len(shap_values)
        # x_test = x_test.reshape(1,x_test.shape[1],666,6)
        # x_test = np.mean(x_test,axis=1)
        # shap_values = np.mean(shap_values,axis=1)


        start_year = 2018
        end_year = 2018
        df = pd.read_csv('D:\ERA-5\Gul wan\coordinates11.csv')
        data = xr.open_dataset('D:\ERA-5\Gul wan/final.nc')

        latitude = data['latitude'].values

        longitude = data['longitude'].values
        tolerance = 1e-6

        # tolerance = 1e-6  # 容差范围
        #
        selected_data = data['swh'].sel(time=slice(f'{start_year}-01-01', f'{end_year}-12-31'))
        swh = selected_data.values[:1, :, :]
        "不可以不copy!!!!! 不然内存会泄露"






        shap_values2 = []
        test2 = []
        for j in range(4):

            swh2 = np.zeros((100*12, len(latitude), len(longitude)))  # 创建新的swh2数组
            swh3 = np.zeros((100*12, len(latitude), len(longitude)))  # 创建新的swh3数组

            ss = shap_values[:,:,j]
            # xx = x_test[:,:,j]
            i = 0
            for lat_idx, lat in enumerate(latitude):

                for lon_idx, lon in enumerate(longitude):

                    # lat = round(lat,2)
                    # lon = round(lon,2)
                    if any((abs(lat - lat_val) < tolerance) and (abs(lon - lon_val) < tolerance) for lat_val, lon_val in
                           zip(df['Latitude'].values, df['Longitude'].values)):

                        swh2[:, lat_idx, lon_idx] = ss[:, i]

                        # print('swh',swh2[:, lat_idx, lon_idx])
                        #
                        #
                        #
                        # print('swh_true',swh3[:, lat_idx, lon_idx])
                        i += 1
                    else:
                        swh2[:, lat_idx, lon_idx] = np.nan


            shap_values2.append(swh2)




        shap_values = np.stack(shap_values2,axis=3) #(batch_Size*sequence,h,w,feature)
        # ss = np.mean(shap_values,axis=1)
        # ss = np.mean(ss,axis=1)
        # print(ss)
        np.save('shap_values_12h503zf', shap_values)

        # print(shap_values.shape)
        #
        # shap_values = shap_values.reshape(shap_values.shape[1],shap_values.shape[2],shap_values.shape[3])
        # test2 = np.stack(test2,axis=3)




        # shap.image_plot(shap_values,test2)

        # shap.summary_plot(shap_values,feature_names=feature_names)







#
# def run_gcrnn(args):
#     with open(args.config_filename) as f:
#         supervisor_config = yaml.safe_load(f)
#
#         graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
#         station_ids, station_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)
#
#         supervisor = GCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)
#         shape_value = supervisor.occlu_sen('test')
#         print(shape_value)
#
#         # mean_score, outputs = supervisor.evaluate2('test')
#         # np.savez_compressed(args.output_filename, **outputs)
#         # print("MAE : {}".format(mean_score))
#         # print('Predictions saved as {}.'.format(args.output_filename))
#
#
if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run Pytorch on cpu.')
    parser.add_argument('--config_filename', default='data/model/pretrained/Heilongjiang_province/config.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--output_filename', default='data/fcrnn_heilongjiang_predictions.npz')
    args = parser.parse_args()
    shap1(args)

