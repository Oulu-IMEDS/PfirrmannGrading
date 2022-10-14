import hydra
import timm
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import MultiStepLR, StepLR, CyclicLR, ReduceLROnPlateau
from src.training.focal_loss import FocalLoss
from src.model.metamodel import MultiModalENet
from src.model.spinenet import GradingModel
from src.model.spinenetv1 import SpineNetV1
from src.model.spinenet_miccai import VertTransformerEncoder
from vit_pytorch import ViT


def build_model(cfg, logger):
    pipeline = list(cfg.mode.keys())[0]
    if pipeline == 'segmentation':
        if cfg.segmentation_architecture['_target_'] == 'segmentation_models_pytorch.Unet':
            # logger.info("Using UNET architecture")
            model = hydra.utils.instantiate(cfg.segmentation_architecture)
        elif cfg.segmentation_architecture['_target_'] == 'segmentation_models_pytorch.UnetPlusPlus':
            # logger.info("Using UnetPlusPlus architecture")
            model = hydra.utils.instantiate(cfg.segmentation_architecture)
        elif cfg.segmentation_architecture['_target_'] == 'segmentation_models_pytorch.FPN':
            # logger.info("Using FPN architecture")
            model = hydra.utils.instantiate(cfg.segmentation_architecture)

        criterion = hydra.utils.instantiate(cfg.training.loss_function)
        optimizer = Adam(model.parameters(), weight_decay=cfg.training.optimizer.weight_decay)
        scheduler = MultiStepLR(optimizer,
                                milestones=[cfg.training.scheduler.step_low, cfg.training.scheduler.step_high],
                                gamma=0.1)
    elif pipeline == 'classification':
        if cfg.classification_architecture['_target_'] == 'torchvision.models.resnet18':
            # logger.info("Using Resnet18 architecture")
            model = hydra.utils.instantiate(cfg.classification_architecture)

            # Changing the input layer to accept 4 channels
            model.conv1 = nn.Conv2d(cfg.mode.classification.num_slices, 64, kernel_size=(7, 7),
                                    stride=(2, 2), padding=(3, 3), bias=False)

            model.fc = nn.Sequential(nn.Linear(512, 256),
                                     nn.Dropout(p=0.1),
                                     nn.Linear(256, 128),
                                     nn.Linear(128, cfg.mode.classification.num_classes))
        elif cfg.classification_architecture['_target_'] == 'torchvision.models.resnet34':
            # logger.info("Using Resnet34 architecture")
            model = hydra.utils.instantiate(cfg.classification_architecture)

            # Changing the input layer to accept 4 channels
            model.conv1 = nn.Conv2d(cfg.mode.classification.num_slices, 64, kernel_size=(7, 7), stride=(2, 2),
                                    padding=(3, 3), bias=False)

            model.fc = nn.Sequential(nn.Linear(512, 256),
                                     nn.Dropout(p=0.1),
                                     nn.Linear(256, 128),
                                     nn.Linear(128, cfg.mode.classification.num_classes)
                                     )
        elif cfg.classification_architecture['_target_'] == 'torchvision.models.resnet50':
            # logger.info("Using Resnet50 architecture")
            model = hydra.utils.instantiate(cfg.classification_architecture)

            # Changing the input layer to accept 4 channels
            model.conv1 = nn.Conv2d(cfg.mode.classification.num_slices, 64, kernel_size=(7, 7), stride=(2, 2),
                                    padding=(3, 3), bias=False)

            model.fc = nn.Sequential(nn.Linear(2048, 256),
                                     nn.Dropout(p=0.1),
                                     nn.Linear(256, 128),
                                     nn.Linear(128, cfg.mode.classification.num_classes)
                                     )
        elif cfg.classification_architecture['_target_'] == 'torchvision.models.efficientnet_b0':
            # logger.info("Using Efficientnet-B0 architecture")
            model = hydra.utils.instantiate(cfg.classification_architecture)

            # Changing the input layer to accept 4 channels
            model.features[0][0] = nn.Conv2d(cfg.mode.classification.num_slices, 32, kernel_size=(3, 3), stride=(2, 2),
                                             padding=(1, 1), bias=False)

            model.classifier = nn.Sequential(nn.Linear(1280, 512),
                                             nn.Dropout(p=0.2),
                                             nn.Linear(512, 256),
                                             nn.Linear(256, cfg.mode.classification.num_classes))
        elif cfg.classification_architecture['_target_'] == 'torchvision.models.efficientnet_b3':
            # logger.info("Using Efficientnet-B0 architecture")
            model = hydra.utils.instantiate(cfg.classification_architecture)

            # Changing the input layer to accept 4 channels
            model.features[0][0] = nn.Conv2d(cfg.mode.classification.num_slices, 40, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            model.classifier = nn.Sequential(nn.Linear(1536, 512),
                                             nn.Dropout(p=0.2),
                                             nn.Linear(512, 256),
                                             nn.Linear(256, cfg.mode.classification.num_classes))
        elif cfg.classification_architecture['_target_'] == 'spinenetv2':
            model = GradingModel()
        elif cfg.classification_architecture['_target_'] == 'spinenetv1':
            model = SpineNetV1()
        elif cfg.classification_architecture['_target_'] == 'spinenet_miccai':
            model = VertTransformerEncoder()
        elif cfg.classification_architecture['_target_'] == 'timm.models.vit_base_patch16_224':
            logger.info("Using VIT architecture")
            # model = hydra.utils.instantiate(cfg.classification_architecture)
            # model.head = nn.Linear(model.head.in_features, cfg.mode.classification.num_classes)
            model = timm.create_model("vit_base_patch16_224", pretrained=True, in_chans=cfg.mode.classification.num_slices)
            for param in model.parameters():
                param.requires_grad = False
            feature_extractor = model.head.in_features
            # last_layer = nn.Linear(num_inputs, cfg.mode.classification.num_classes)
            last_layer = nn.Sequential(nn.Linear(feature_extractor, 512),
                                       nn.Dropout(p=0.2),
                                       nn.Linear(512, 256),
                                       nn.Linear(256, cfg.mode.classification.num_classes))
            model.head = last_layer
            # print(model)
        if cfg.training.scheduler.type == 'step':
            logger.info('Using StepLR scheduler')
            if cfg.classification_architecture['_target_'] == 'spinenetv2':
                logger.info('Using Adam Optimizer with 1e-4')
                optimizer = Adam(model.parameters(), lr=1e-4)
                scheduler = None
            elif cfg.classification_architecture['_target_'] == 'spinenetv1':
                logger.info('Using Adam Optimizer with 1e-6')
                optimizer = Adam(model.parameters(), lr=1e-6, weight_decay=0.0005)
                scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)
                #scheduler = None
            elif cfg.classification_architecture['_target_'] == 'spinenet_miccai':
                logger.info('Using Spinenet Miccai')
                optimizer = Adam(model.parameters(), lr=1e-4)
                scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)
            else:
                optimizer = Adam(model.parameters(), lr=cfg.training.learning_rate,
                                  weight_decay=cfg.training.optimizer.weight_decay)
                scheduler = StepLR(optimizer, step_size=cfg.training.scheduler.step, gamma=cfg.training.scheduler.gamma)
        elif cfg.training.scheduler.type == 'cyclic':
            logger.info('Using CyclicLR scheduler')
            optimizer = SGD(model.parameters(), lr=cfg.training.learning_rate, momentum=0.9)
            scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=cfg.training.learning_rate)

        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)
    elif pipeline == 'multimodal':
        model = MultiModalENet(cfg)

        if cfg.training.scheduler.type == 'step':
            logger.info('Using StepLR scheduler')
            optimizer = AdamW(model.parameters(), lr=cfg.training.learning_rate,
                              weight_decay=cfg.training.optimizer.weight_decay)
            scheduler = StepLR(optimizer, step_size=cfg.training.scheduler.step, gamma=cfg.training.scheduler.gamma)
        elif cfg.training.scheduler.type == 'cyclic':
            logger.info('Using CyclicLR scheduler')
            optimizer = SGD(model.parameters(), lr=cfg.training.learning_rate, momentum=0.9)
            scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=cfg.training.learning_rate)

        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)

    return model, criterion, optimizer, scheduler
