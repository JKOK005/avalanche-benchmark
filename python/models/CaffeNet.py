import torch
from torch import nn

class CaffeNet(nn.Module):
	def __init__(self, num_classes = 10):
		super(CaffeNet, self).__init__()

		self.features = nn.Sequential(
							nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
							nn.ReLU(inplace=True),
							nn.MaxPool2d(kernel_size=3, stride=2),
							nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75),
							# conv 2
							nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
							nn.ReLU(inplace=True),
							nn.MaxPool2d(kernel_size=3, stride=2),
							nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75),
							# conv 3
							nn.Conv2d(256, 384, kernel_size=3, padding=1),
							nn.ReLU(inplace=True),
							# conv 4
							nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
							nn.ReLU(inplace=True),
							# conv 5
							nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
							nn.ReLU(inplace=True),
							nn.MaxPool2d(kernel_size=3, stride=2)
						)

		self.classifier = nn.Sequential(
							nn.Linear(1024, 4096),
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