import gzip
import os
import pickle
import string
from typing import Optional

from iflearner.datasets.fl_dataset import FLDateset
from iflearner.datasets.utils import *


class MNIST(FLDateset):
    resources = [
        (
            "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
            "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        ),
        (
            "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
            "d53e105ee54ea40749a09fcbcd1e9432",
        ),
        (
            "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
            "9fb629c4189551a2d022fa330f9573f3",
        ),
        (
            "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
            "ec29112dd5afa0611ce80d1b7f02629c",
        ),
    ]
    training_file = "training.pt"
    test_file = "test.pt"
    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

    def __init__(self, root: str, download: bool = False):
        super().__init__()
        self.root = root

        self.raw_folder = os.path.join(self.root, self.__class__.__name__, "raw")
        self.processed_folder = os.path.join(
            self.root, self.__class__.__name__, "processed"
        )
        if download:
            self.download()

        with open(os.path.join(self.processed_folder, self.training_file), "rb") as f:
            self.train_x, self.train_targets = pickle.load(f)
        with open(os.path.join(self.processed_folder, self.test_file), "rb") as f:
            self.test_x, self.test_targets = pickle.load(f)

    def _check_exists(self) -> bool:
        return os.path.exists(
            os.path.join(self.processed_folder, self.training_file)
        ) and os.path.exists(os.path.join(self.processed_folder, self.test_file))

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition("/")[2]
            download_and_extract_archive(
                url, download_root=self.raw_folder, filename=filename, md5=md5
            )

        train_set = (
            read_image_file(os.path.join(self.raw_folder, "train-images-idx3-ubyte")),
            read_label_file(os.path.join(self.raw_folder, "train-labels-idx1-ubyte")),
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, "t10k-images-idx3-ubyte")),
            read_label_file(os.path.join(self.raw_folder, "t10k-labels-idx1-ubyte")),
        )

        with open(os.path.join(self.processed_folder, self.training_file), "wb") as f:
            pickle.dump(train_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), "wb") as f:
            pickle.dump(test_set, f)


class FashionMNIST(MNIST):
    resources = [
        (
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
            "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
        ),
        (
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
            "25c81989df183df01b3e8a0aad5dffbe",
        ),
        (
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
            "bef4ecab320f06d8554ea6380940ec79",
        ),
        (
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
            "bb300cfdad3c16e7a12a480ee83cd310",
        ),
    ]
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]


class KMNIST(MNIST):
    resources = [
        (
            "http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz",
            "bdb82020997e1d708af4cf47b453dcf7",
        ),
        (
            "http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz",
            "e144d726b3acfaa3e44228e80efcd344",
        ),
        (
            "http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz",
            "5c965bf0a639b31b8f53240b1b52f4d7",
        ),
        (
            "http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz",
            "7320c461ea6c1c855c0b718fb2a4b134",
        ),
    ]
    classes = ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"]


class EMNIST(MNIST):
    """`EMNIST <https://www.westernsydney.edu.au/bens/home/reproducible_research/emnist>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``EMNIST/processed/training.pt``
            and  ``EMNIST/processed/test.pt`` exist.
        split (string): The dataset has 6 different splits: ``byclass``, ``bymerge``,
            ``balanced``, ``letters``, ``digits`` and ``mnist``. This argument specifies
            which one to use.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    # Updated URL from https://www.nist.gov/node/1298471/emnist-dataset since the
    # _official_ download link
    # https://cloudstor.aarnet.edu.au/plus/s/ZNmuFiuQTqZlu9W/download
    # is (currently) unavailable
    url = "http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip"
    md5 = "58c8d27c78d21e728a6bc7b3cc06412e"
    splits = ("byclass", "bymerge", "balanced", "letters", "digits", "mnist")
    # Merged Classes assumes Same structure for both uppercase and lowercase version
    _merged_classes = {
        "C",
        "I",
        "J",
        "K",
        "L",
        "M",
        "O",
        "P",
        "S",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
    }
    _all_classes = set(list(string.digits + string.ascii_letters))
    classes_split_dict = {
        "byclass": list(_all_classes),
        "bymerge": sorted(list(_all_classes - _merged_classes)),
        "balanced": sorted(list(_all_classes - _merged_classes)),
        "letters": list(string.ascii_lowercase),
        "digits": list(string.digits),
        "mnist": list(string.digits),
    }

    def __init__(self, root: str, split: str = "mnist", **kwargs: Any) -> None:
        self.split = verify_str_arg(split, "split", self.splits)
        self.training_file = self._training_file(split)
        self.test_file = self._test_file(split)
        super(EMNIST, self).__init__(root, **kwargs)
        self.classes = self.classes_split_dict[self.split]

    @staticmethod
    def _training_file(split) -> str:
        return "training_{}.pt".format(split)

    @staticmethod
    def _test_file(split) -> str:
        return "test_{}.pt".format(split)

    def download(self) -> None:
        """Download the EMNIST data if it doesn't exist in processed_folder already."""
        import shutil

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        print("Downloading and extracting zip archive")
        download_and_extract_archive(
            self.url,
            download_root=self.raw_folder,
            filename="emnist.zip",
            remove_finished=True,
            md5=self.md5,
        )
        gzip_folder = os.path.join(self.raw_folder, "gzip")
        for gzip_file in os.listdir(gzip_folder):
            if gzip_file.endswith(".gz"):
                extract_archive(os.path.join(gzip_folder, gzip_file), gzip_folder)

        # process and save as torch files
        for split in self.splits:
            print("Processing " + split)
            training_set = (
                read_image_file(
                    os.path.join(
                        gzip_folder, "emnist-{}-train-images-idx3-ubyte".format(split)
                    )
                ),
                read_label_file(
                    os.path.join(
                        gzip_folder, "emnist-{}-train-labels-idx1-ubyte".format(split)
                    )
                ),
            )
            test_set = (
                read_image_file(
                    os.path.join(
                        gzip_folder, "emnist-{}-test-images-idx3-ubyte".format(split)
                    )
                ),
                read_label_file(
                    os.path.join(
                        gzip_folder, "emnist-{}-test-labels-idx1-ubyte".format(split)
                    )
                ),
            )
            with open(
                os.path.join(self.processed_folder, self._training_file(split)), "wb"
            ) as f:
                pickle.dump(training_set, f)

            with open(
                os.path.join(self.processed_folder, self._test_file(split)), "wb"
            ) as f:
                pickle.dump(test_set, f)
        shutil.rmtree(gzip_folder)

        print("Done!")


if __name__ == "__main__":
    d = EMNIST("./data", download=True)
    print(d.train_data.shape)
