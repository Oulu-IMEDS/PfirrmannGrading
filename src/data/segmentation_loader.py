import os
import logging
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch

log = logging.getLogger(__name__)


class SegmentationLoader(Dataset):
    def __init__(self, cfg, data_slice, aug_type='train'):
        # Hydra Configuration
        self.cfg = cfg
        # Data slice of overall dataset
        self.data_slice = data_slice
        # Size of the Data Slice
        self.dataset_size = data_slice.shape[0]
        # Apply transformations based on train or val set
        self.aug_type = aug_type

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        # Get the filename of the image
        img_file = self.data_slice.iloc[index]['image']
        img_raw = cv2.imread(os.path.join(self.cfg.data.segmentation.dir, self.cfg.data.segmentation.images, img_file))
        # Masks should not have channel. If you include the channel if you get an error so [:,:,0] is added
        img_mask = cv2.imread(os.path.join(self.cfg.data.segmentation.dir, self.cfg.data.segmentation.masks, img_file))[:, :, 0]
        transform_raw, transform_mask = self.apply_img_transforms(img_raw, img_mask)
        return {"img_file": img_file, "transformed_raw": transform_raw, "transformed_mask": transform_mask}

    # Apply Image Transformations using Albumentations
    # When you normalize the class labels of masks change.
    def apply_img_transforms(self, img_raw, img_mask):
        if self.aug_type == 'train':
            train_transform = A.Compose([A.Resize(512, 512), ToTensorV2()])
            transformed_img = train_transform(image=img_raw)['image']
            transformed_mask = (train_transform(image=img_mask)['image']).long()
            # Tensors are to be converted to float tensors as cv2 is giving byte tensor
            transformed_img = transformed_img.type(torch.FloatTensor)
            return transformed_img, transformed_mask
        elif self.aug_type == 'val':
            val_transform = A.Compose([A.Resize(512, 512), ToTensorV2()])
            transformed_img = val_transform(image=img_raw)['image']
            transformed_mask = (val_transform(image=img_mask)['image']).long()
            # Tensors are to be converted to float tensors as cv2 is giving byte tensor
            transformed_img = transformed_img.type(torch.FloatTensor)
            return transformed_img, transformed_mask
        else:
            test_transform = A.Compose([A.Resize(512, 512), ToTensorV2()])
            transformed_img = test_transform(image=img_raw)['image']
            transformed_mask = (test_transform(image=img_mask)['image']).long()
            # Tensors are to be converted to float tensors as cv2 is giving byte tensor
            transformed_img = transformed_img.type(torch.FloatTensor)
            return transformed_img, transformed_mask
