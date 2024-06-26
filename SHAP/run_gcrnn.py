import argparse
import numpy as np
import os
import sys
import yaml

from lib.utils import load_graph_data
from model.gcrnn_supervisor import GCRNNSupervisor


def run_gcrnn(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.safe_load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        station_ids, station_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        supervisor = GCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)
        mean_score, outputs = supervisor.evaluate2('test')
        np.savez_compressed(args.output_filename, **outputs)
        print("MAE : {}".format(mean_score))
        print('Predictions saved as {}.'.format(args.output_filename))


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run Pytorch on cpu.')
    parser.add_argument('--config_filename', default='data/model/pretrained/Heilongjiang_province/config.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--output_filename', default='data/fcrnn_heilongjiang_predictions.npz')
    args = parser.parse_args()
    run_gcrnn(args)

