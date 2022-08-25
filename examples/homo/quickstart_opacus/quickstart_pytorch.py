import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from torch import nn
from torchvision import datasets, transforms

from iflearner.business.homo.argument import parser
from iflearner.business.homo.pytorch_trainer import PyTorchTrainer
from iflearner.business.homo.train_client import Controller


class Model(nn.Module):
    def __init__(self, num_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Mnist(PyTorchTrainer):
    def __init__(self, lr=0.15, momentum=0.5, delta=1e-5) -> None:
        self._lr = lr
        self._delta = delta
        self._device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print(f"device: {self._device}")
        model = Model(num_channels=1, num_classes=10).to(self._device)

        super().__init__(model)

        optimizer = optim.SGD(self._model.parameters(), lr=lr, momentum=momentum)
        self._loss = F.nll_loss

        apply_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_dataset = datasets.MNIST(
            "./data", train=True, download=True, transform=apply_transform
        )
        test_dataset = datasets.MNIST(
            "./data", train=False, download=True, transform=apply_transform
        )
        train_data = torch.utils.data.DataLoader(
            train_dataset, batch_size=64, shuffle=True
        )
        self._test_data = torch.utils.data.DataLoader(
            test_dataset, batch_size=64, shuffle=False
        )
        self._privacy_engine = PrivacyEngine()
        (
            self._model,
            self._optimizer,
            self._train_data,
        ) = self._privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_data,
            noise_multiplier=1.1,
            max_grad_norm=1.0,
        )

    def fit(self, epoch):
        self._model.to(self._device)
        self._model.train()
        print(
            f"Epoch: {epoch}, the size of training dataset: {len(self._train_data.dataset)}, batch size: {len(self._train_data)}"
        )
        losses = []
        for batch_idx, (data, target) in enumerate(self._train_data):
            data, target = data.to(self._device), target.to(self._device)
            self._optimizer.zero_grad()
            output = self._model(data)
            loss = self._loss(output, target)
            loss.backward()
            self._optimizer.step()
            losses.append(loss.item())

        epsilon, best_alpha = self._privacy_engine.accountant.get_privacy_spent(
            delta=self._delta
        )
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"(ε = {epsilon:.2f}, δ = {self._delta}) for α = {best_alpha}"
        )

    def evaluate(self, epoch):
        self._model.to(self._device)
        self._model.eval()
        test_loss = 0
        correct = 0
        print(f"The size of testing dataset: {len(self._test_data.dataset)}")
        with torch.no_grad():
            for data, target in self._test_data:
                data, target = data.to(self._device), target.to(self._device)
                output = self._model(data)
                test_loss += self._loss(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self._test_data.dataset)

        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                test_loss,
                correct,
                len(self._test_data.dataset),
                100.0 * correct / len(self._test_data.dataset),
            )
        )

        return {"loss": test_loss, "acc": correct / len(self._test_data.dataset)}


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    mnist = Mnist()
    controller = Controller(args, mnist)
    controller.run()
