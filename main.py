from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml
import pickle
import numpy as np
from lib.utils import load_graph_data
from model.pytorch.supervisor import Supervisor


def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.safe_load(f)
        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)
        with open(supervisor_config['data'].get('graph_list_filename'), 'rb') as f:
            adj_0, adj_1 = pickle.load(f)
        supervisor = Supervisor(adj_mx0=adj_0, adj_mx=adj_1, **supervisor_config)

        supervisor.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='data/model/GMSDR_LA.yaml', type=str,
                        help='Configuration filename for restoring the model.')
    args = parser.parse_args()
    main(args)
