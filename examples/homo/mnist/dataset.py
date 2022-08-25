import copy

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def get_data(num_clients: int, client_idx: int, iid: bool):
    train_data, test_data, client_groups = get_dataset(
        num_clients=num_clients,
        data_dir="../data/mnist",
        is_iid=iid,
        noiid_isunequal=False,
    )

    batch_size = 64
    idxs = client_groups[client_idx]

    idxs_train = idxs[: int(0.8 * len(idxs))]
    idxs_val = idxs[int(0.8 * len(idxs)) :]

    train_loader = torch.utils.data.DataLoader(
        DatasetSplit(train_data, idxs_train), batch_size=batch_size, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        DatasetSplit(train_data, idxs_val), batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader, valid_loader


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class."""

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


def get_dataset(
    num_clients: int,
    data_dir: str = "./data/mnist",
    is_iid: bool = True,
    noiid_isunequal: bool = False,
):
    apply_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=apply_transform
    )

    test_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=apply_transform
    )

    if is_iid:
        # Sample IID client data from Mnist
        client_groups = mnist_iid(train_dataset, num_clients)
    else:
        # Sample Non-IID client data from Mnist
        if noiid_isunequal:
            # Chose uneuqal splits for every client
            client_groups = mnist_noniid_unequal(train_dataset, num_clients)
        else:
            # Chose euqal splits for every client
            client_groups = mnist_noniid(train_dataset, num_clients)
    return train_dataset, test_dataset, client_groups


def mnist_iid(dataset, num_client) -> dict:
    """Sample I.I.D.

    client data from MNIST dataset
    Args:
        dataset(list): mnist dataset.
        num_client(int): The num of clients.
    Returns:
        dict of image index
    """
    num_items = int(len(dataset) / num_client)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_client):
        dict_users[i] = list(set(np.random.choice(all_idxs, num_items, replace=False)))
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    return dict_users


def mnist_noniid(dataset, num_client):
    """
    Sample non-I.I.D client data from MNIST dataset
    Args:
        dataset(list): mnist dataset.
        num_client(int): The num of clients.
    Returns:
        dict of image index
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_client)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_client):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]), axis=0
            )
    return dict_users


def mnist_noniid_unequal(dataset, num_users) -> dict:
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    Args:
        dataset(list): mnist dataset.
        num_client(int): The num of clients.
    Returns:
        a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard + 1, size=num_users)
    random_shard_size = np.around(
        random_shard_size / sum(random_shard_size) * num_shards
    )
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]),
                    axis=0,
                )

        random_shard_size = random_shard_size - 1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]),
                    axis=0,
                )
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]),
                    axis=0,
                )

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand * num_imgs : (rand + 1) * num_imgs]),
                    axis=0,
                )

    return dict_users


if __name__ == "__main__":
    num_clients = 3
    train_dataset, test_dataset, client_groups = get_dataset(
        num_clients=num_clients,
        data_dir="./data/mnist",
        is_iid=False,
        noiid_isunequal=True,
    )
    print("client_group is:%s \n" % len(client_groups))
    total = 0
    for i in range(num_clients):
        print("client_goup %s len is:%s" % (i, len(client_groups[i])))
        total += len(client_groups[i])
    print("total num is:%s" % (total))
