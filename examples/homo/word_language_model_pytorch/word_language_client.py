import torch
import torch.nn.functional as F
import torch.optim as optim
from data import Corpus
from model import TransformerModel

from iflearner.business.homo.argument import parser
from iflearner.business.homo.pytorch_trainer import PyTorchTrainer
from iflearner.business.homo.train_client import Controller


class WordLanguage(PyTorchTrainer):
    def __init__(
        self,
        lr=0.02,
        momentum=0.5,
        data_path="./data/wikitext-2",
        bptt=35,
        client_name="client",
    ) -> None:
        self._lr = lr
        self._bptt = 35
        self._client_name = client_name
        self._device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print(f"device: {self._device}")

        corpus = Corpus(data_path)

        if client_name == "client1":
            self._train_data = self.batchify(corpus.train, 128)
        if client_name == "client2":
            self._train_data = self.batchify(corpus.valid, 64)
        self._test_data = self.batchify(corpus.test, 64)

        self.ntokens = len(corpus.dictionary)
        self._model = TransformerModel(self.ntokens, 200, 2, 200, 2, 0.2).to(
            self._device
        )

        super().__init__(self._model)

        self._optimizer = optim.SGD(self._model.parameters(), lr=lr, momentum=momentum)
        self._loss = F.nll_loss

        self._best_val_loss = None

    def get_batch(self, source, i):
        seq_len = min(self._bptt, len(source) - 1 - i)
        data = source[i : i + seq_len]
        target = source[i + 1 : i + 1 + seq_len].view(-1)
        return data, target

    def batchify(self, data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(self._device)

    def fit(self, epoch):
        self._model.to(self._device)
        self._model.train()
        print(f"Epoch: {epoch}")

        for batch, i in enumerate(range(0, self._train_data.size(0) - 1, self._bptt)):
            data, targets = self.get_batch(self._train_data, i)
            self._model.zero_grad()
            output = self._model(data)
            output = output.view(-1, self.ntokens)

            loss = self._loss(output, targets)
            loss.backward()
            if batch % 100 == 0:
                print(f"{self._client_name} batch {batch} loss: ", loss.item())

            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.25)
            for p in self._model.parameters():
                p.data.add_(p.grad, alpha=-self._lr)

    def evaluate(self, epoch):
        self._model.to(self._device)
        self._model.eval()
        test_loss = 0

        with torch.no_grad():
            for i in range(0, self._test_data.size(0) - 1, self._bptt):
                data, targets = self.get_batch(self._test_data, i)
                output = self._model(data)
                output = output.view(-1, self.ntokens)

                test_loss += len(data) * self._loss(output, targets).item()

        test_loss /= len(self._test_data) - 1

        if not self._best_val_loss or test_loss < self._best_val_loss:
            self._best_val_loss = test_loss
        else:
            self._lr /= 4
        print("Test set: Average loss: {:.4f},".format(test_loss))

        return {"loss": test_loss}


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    wl = WordLanguage(client_name=args.name)
    controller = Controller(args, wl)
    controller.run()
