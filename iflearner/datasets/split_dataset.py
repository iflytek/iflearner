import os.path

import numpy as np

from iflearner.datasets.sampler import Sampler
from iflearner.datasets.fl_dataset import FLDateset
from iflearner.datasets.utils import read_yaml
import argparse
from iflearner.datasets import *

SUPPORT_DATASET = ['MNIST', 'FashionMNIST',
                   'KMNIST', 'EMNIST', 'CIFAR10', 'CIFAR100']


def read_yaml_file(file_path):
    with open(file_path, mode='r', encoding='utf-8') as fd:
        data = yaml.load(fd, Loader=yaml.FullLoader)
    return data


def main(args):
    if args.config is not None:
        args = read_yaml_file(args.config)
        args = argparse.Namespace(**args)
    print(args)

    if args.dataset:
        if args.dataset not in SUPPORT_DATASET:
            raise Exception(f'only support {", ".join(SUPPORT_DATASET)}')

        dataset = eval(args.dataset)('./data', True)
        data = np.array(dataset.train_data)
        label = np.array(dataset.train_labels)
        if args.save_test_set_path:
            if not os.path.exists(args.save_test_set_path):
                os.makedirs(args.save_test_set_path)
            test_data = np.array(dataset.test_data)
            np.save(os.path.join(args.save_test_set_path, 'test_data'), test_data)
            test_label = np.array(dataset.test_labels)
            np.save(os.path.join(args.save_test_set_path,
                    'test_label'), test_label)

    else:
        if args.data_path is None or args.label_path is None:
            raise Exception(f'A dataset must be set!')

        data = np.load(args.data_path)
        label = np.load(args.label_path)
        assert len(data) == len(label)
    clients = args.clients
    if not clients:
        raise Exception
    if args.iid:
        sampler = Sampler(label, clients, 'iid')
    else:
        sampler = Sampler(label, clients, args.noniid)

    client_index = sampler.get_client_index
    for name, indexes in client_index.items():
        print(name)
        indexes = list(indexes)
        target = label[indexes]
        X = data[indexes]
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        np.save(os.path.join(args.save_path, f'{name}-train_data'), X)
        np.save(os.path.join(args.save_path, f'{name}-train_label'), target)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--config', type=str, help='The path of config file')
    parse.add_argument('--iid', type=bool, default=False,
                       help='if to split dataset as iid', )
    parse.add_argument('--noniid', type=str, default='noniid',
                       help='the kind of noniid type', )
    parse.add_argument('--alpha', type=float, default=1,
                       help='the parameter to control the noniid degree, only using when iid is false')
    parse.add_argument('--save_path', type=str, default='./data',
                       help='the path of saving the splitted dataset')
    parse.add_argument('--dataset', type=str, default='MNIST',
                       help='the name of dataset')
    parse.add_argument('--data_path', type=str,
                       help='the whole data in .npy format')
    parse.add_argument('--label_path', type=str,
                       help='the whole label in .npy format')
    parse.add_argument('--save_test_set_path', type=bool,
                       default=False, help='if save the test dataset in')
    parse.add_argument('--clients', type=str, nargs='+',
                       default=['client2', 'client1'], help='clients names, eg. client1 client2')

    args = parse.parse_args()
    # print(type(args))
    main(args)
