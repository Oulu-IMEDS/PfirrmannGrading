import torch
import torch.nn as nn


class SpineNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=96, kernel_size=(7, 7), stride=(2, 2), padding=3)
            , nn.ReLU()
            , nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            , nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding=2)
            , nn.ReLU()
            , nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            , nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1)
            , nn.ReLU()
            , nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1)
            , nn.ReLU()
            , nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            , nn.Flatten()
        )
        self.classifier = nn.Sequential(nn.Linear(in_features=50176, out_features=1024)
                                        , nn.Linear(in_features=1024, out_features=1024)
                                        , nn.Linear(in_features=1024, out_features=5)
                                        , nn.Softmax(dim=-1))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
