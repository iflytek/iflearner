## How to use split_dataset

Run ``python iflearner/datasets/split_dataset.py -h`` to see the help of the command.

```
optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       The path of config file
  --iid IID             if to split dataset as iid
  --noniid NONIID       the kind of noniid type
  --alpha ALPHA         the parameter to control the noniid degree, only using when iid is false
  --save_path SAVE_PATH
                        the path of saving the splitted dataset
  --dataset DATASET     the name of dataset
  --data_path DATA_PATH
                        the whole data in .npy format, only using when dataset is None
  --label_path LABEL_PATH
                        the whole label in .npy format, only using when dataset is None
  --save_test_set_path SAVE_TEST_SET_PATH
                        if save the test dataset in
  --clients CLIENTS [CLIENTS ...]
                        clients names, eg. client1 client2
```

If the `config` argument is not None, all other argument must be set in the config file with ``yaml`` formart. It is like:

```yaml
iid: False
noniid: "noniid"
alpha: 1
save_path: "data_train"
dataset: "MNIST"
data_path: "/data/hanyuhu/iflearber-github/data/0-train_data.npy"
label_path: "/data/hanyuhu/iflearber-github/data/0-train_label.npy"
save_test_set_path: "data_test"
clients:
  - client1
  - client2
```

