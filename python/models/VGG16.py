import torch
from torch import nn

class Vgg16(nn.Module):
    def __init__(self, num_classes=10):
        super(Vgg16, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding="same"),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding="same"),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),

            nn.Conv2d(64, 128, kernel_size=3, padding="same"),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding="same"),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),

            nn.Conv2d(128, 256, kernel_size=3, padding="same"),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding="same"),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding="same"),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),

            nn.Conv2d(256, 512, kernel_size=3, padding="same"),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding="same"),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding="same"),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),

            nn.Conv2d(512, 512, kernel_size=3, padding="same"),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding="same"),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding="same"),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return torch.nn.functional.softmax(x)