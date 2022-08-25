from collections import defaultdict

import numpy as np


class Sampler:
    def __init__(self, label: list, clients: list, method="iid"):
        self.targets = np.array(label, dtype="int64")
        self.clients = clients

        if method == "iid":
            self.client_index = self.iid()
        elif method == "noniid":
            self.client_index = self.noniid()
        else:
            pass

    def iid(self):
        clients_index = defaultdict(set)
        length = len(self.targets)
        clients_num = len(self.clients)
        all_idxs = np.arange(length)
        for i in range(clients_num):
            clients_index[self.clients[i]] = set(
                np.random.choice(all_idxs, length // clients_num, replace=False)
            )
            all_idxs = list(set(all_idxs) - clients_index[i])

        return clients_index

    def noniid(self):
        clients_num = len(self.clients)
        num_shards, num_imgs = clients_num * 15, len(self.targets) // (clients_num * 15)
        idx_shard = [i for i in range(num_shards)]
        clients_index = {i: np.array([], dtype="int64") for i in range(clients_num)}
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
                        clients_index[i] = np.concatenate(
                            (
                                clients_index[i],
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
    s = Sampler(data.train_labels, ["1", "2", "3"], "noniid")
    index = s.client_index
    print(index)
    print(s.targets)
    for name, indexes in s.client_index.items():
        print(s.targets[indexes])

    df = pd.DataFrame(
        {name: pd.Series(s.targets[index]) for name, index in s.client_index.items()}
    )
    for col in df.columns:
        print(df[col].value_counts())
