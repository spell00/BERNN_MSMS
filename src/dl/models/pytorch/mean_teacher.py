import pandas as pd
from torch import nn
import queue

class Cifar10CnnModel(nn.Module):
    def __init__(self, n_classes, n_batches=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, n_classes)
        )

        self.dann_discriminator = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, n_batches)
        )
        self.random_init()

    def forward(self, xb):
        x = self.network(xb)
        return x

    def random_init(self, init_func=nn.init.kaiming_uniform_):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class MnistCnnModel(nn.Module):
    def __init__(self, n_classes, n_batches=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),  # output: 64 x 16 x 16

            nn.Conv2d(64, 64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),  # output: 128 x 8 x 8

            nn.Conv2d(64, 128, kernel_size=5, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2 * 2, 3072),
            nn.ReLU(),
            nn.Linear(3072, 2048),
            nn.ReLU(),
            nn.Linear(2048, n_classes)
        )

        self.dann_discriminator = nn.Sequential(
            nn.Linear(128 * 2 * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, n_batches)
        )
        self.random_init()

    def forward(self, xb):
        x = self.network(xb)
        return x

    def random_init(self, init_func=nn.init.kaiming_uniform_):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class MS1CnnModel(nn.Module):
    def __init__(self, n_classes, n_batches=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Dropout2d(),
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.Dropout2d(),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(),
            nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            # nn.Dropout2d(),
            # nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4
            # nn.Flatten(),
            # nn.Linear(64 * 1 * 1, 32),
            # nn.Dropout2d(),
            # nn.ReLU(),
            # nn.Linear(32, 16),
            # nn.Dropout2d(),
            # nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, n_classes)
        )

        self.dann_discriminator = nn.Sequential(
            # nn.Linear(256 * 4 * 4, 1024),
            # nn.ReLU(),
            # nn.Linear(1024, 512),
            # nn.ReLU(),
            nn.Linear(64, n_batches)
        )
        self.random_init()

    def forward(self, xb):
        x = self.network(xb)
        return x

    def random_init(self, init_func=nn.init.kaiming_uniform_):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class MeanTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.models = queue.LifoQueue()
        self.teacher = None

    def add_student(self, model):
        self.models.put(model)

    def update_ema_variables(model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
