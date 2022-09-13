from collections import defaultdict
from ctypes import Union

import numpy as np

from iflearner.datasets.utils import partition_class_samples_with_dirichlet_distribution


class Sampler:
    def __init__(self, label, clients: list, method="iid", **kwargs):
        self.targets = np.array(label, dtype="int64")
        self.clients = clients
        if method == "iid":
            self.client_index = self.iid()
        elif method == "noniid":
            self.client_index = self.noniid()
        elif method == "dirichlet":
            if "alpha" in kwargs:
                self.client_index = self.dirichlet_distribution_non_iid(
                    kwargs.get("alpha")
                )
            else:
                self.client_index = self.dirichlet_distribution_non_iid(1)
    @property
    def get_client_index(self):
        return self.client_index

    def iid(self):
        clients_index = defaultdict(set)
        length = len(self.targets)
        clients_num = len(self.clients)
        all_idxs = np.arange(length)
        for i in self.clients:
            clients_index[i] = set(
                np.random.choice(all_idxs, length // clients_num, replace=False)
            )
            all_idxs = list(set(all_idxs) - clients_index[i])

        return clients_index

    def dirichlet_distribution_non_iid(self, alpha):
        clients_index = defaultdict(set)

        N = len(self.targets)
        K = len(set(self.targets))
        client_num = len(self.clients)
        idx_batch = [[] for _ in range(client_num)]
        for k in range(K):
            # get a list of batch indexes which are belong to label k
            idx_k = np.where(self.targets == k)[0]

            idx_batch = partition_class_samples_with_dirichlet_distribution(
                N, alpha, client_num, idx_batch, idx_k
            )
        for i in range(client_num):
            np.random.shuffle(idx_batch[i])
            clients_index[self.clients[i]] = set(idx_batch[i])
        return clients_index

    def noniid(self):
        clients_num = len(self.clients)
        num_shards, num_imgs = clients_num * 15, len(self.targets) // (clients_num * 15)
        idx_shard = [i for i in range(num_shards)]
        clients_index = {i: np.array([], dtype="int64") for i in self.clients}
        idxs = np.arange(len(self.targets))
        labels = np.array(self.targets)

        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        while len(idx_shard) > 2:
            try:
                for i in range(clients_num):
                    rand_set = set(np.random.choice(idx_shard, 2, replace=False))
                    idx_shard = list(set(idx_shard) - rand_set)
                    for rand in rand_set:
                        clients_index[self.clients[i]] = np.concatenate(
                            (
                                clients_index[self.clients[i]],
                                idxs[rand * num_imgs : (rand + 1) * num_imgs],
                            ),
                            axis=0,
                        )
            except:
                pass
        return clients_index


if __name__ == "__main__":
    import pandas as pd
    from mnist import MNIST

    data = MNIST("./data", True)
    s = Sampler(data.train_labels, ["1", "2", "3"], "dirichlet")
    d = {}
    for name, indexes in s.client_index.items():
        # print(type(s.targets[indexes]))
        indexes = list(indexes)
        d[name] = pd.Series(s.targets[indexes])
        print(s.targets[indexes])

    df = pd.DataFrame(d)
    for col in df.columns:
        print(df[col].value_counts())
