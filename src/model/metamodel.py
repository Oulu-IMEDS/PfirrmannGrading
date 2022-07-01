import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0


# Code inspired from https://www.kaggle.com/code/nroman/melanoma-pytorch-starter-efficientnet
class MultiModalENet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        meta_features_size = len(cfg.mode.multimodal.meta_features.split(","))
        self.model = efficientnet_b0(pretrained=True)
        self.model.features[0][0] = nn.Conv2d(cfg.mode.multimodal.num_slices, 32, kernel_size=(3, 3), stride=(2, 2),
                                              padding=(1, 1), bias=False)
        self.model.classifier = nn.Linear(in_features=1280, out_features=512, bias=True)  # Convnet Output 512

        self.meta_data = nn.Sequential(nn.Linear(meta_features_size, 512),
                                       nn.BatchNorm1d(512),
                                       nn.LeakyReLU(),
                                       nn.Dropout(p=0.2),
                                       nn.Linear(512, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 128),  # FC Output 128
                                       )

        # Add the outputs of Conv Features (512) and Meta Features (128)
        self.output = nn.Sequential(nn.BatchNorm1d(640),
                                    nn.Linear(640, 256),
                                    nn.LeakyReLU(),
                                    nn.Dropout(p=0.2),
                                    nn.Linear(256, cfg.mode.multimodal.num_classes)
                                    )

    def forward(self, inputs):
        images = inputs['images'].to('cuda')
        metadata = inputs['meta_data'].to('cuda')

        cnn_features = self.model(images)
        meta_features = self.meta_data(metadata)
        features = torch.cat((cnn_features, meta_features), dim=1)
        output = self.output(features)
        return output
