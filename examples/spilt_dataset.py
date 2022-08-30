import pandas as pd

from iflearner.datasets.mnist import MNIST
from iflearner.datasets.sampler import Sampler


clients = ['party1', 'party2', 'party3']
dataset = MNIST('./data', True)
sampler = Sampler(dataset.train_labels, clients, 'dirichlet', alpha=2)
clients_index = sampler.client_index

d = {}
for name, index in clients_index.items():
    index = list(index)
    p = pd.Series(dataset.train_labels[index].astype('int64'))

    print(name+'各个类别：')
    print(p.value_counts())
    print()