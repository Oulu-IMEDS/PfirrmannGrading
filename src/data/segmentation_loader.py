import albumentations as A
import cv2
import logging
import os
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

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
        img_raw = cv2.imread(os.path.join(self.cfg.mode.input.dir, self.cfg.mode.input.images, img_file))
        # Masks should not have channel. If you include the channel if you get an error so [:,:,0] is added
        img_mask = cv2.imread(os.path.join(self.cfg.mode.input.dir, self.cfg.mode.input.masks, img_file))[
                   :, :, 0]

        transform_raw, transform_mask = self.apply_img_transforms(img_raw, img_mask)
        return {"img_file": img_file, "transformed_raw": transform_raw, "transformed_mask": transform_mask}

    # Apply Image Transformations using Albumentations
    # When you normalize the class labels of masks change.
    def apply_img_transforms(self, img_raw, img_mask):
        if self.aug_type == 'train':
            train_transform = A.Compose([A.OneOf([A.OpticalDistortion(distort_limit=2, shift_limit=0.5)
                                                     , A.ElasticTransform(alpha=120, sigma=120 * 0.05,
                                                                          alpha_affine=120 * 0.03)
                                                     , A.Affine(rotate=(-10, 10), shear=(-10, 10))
                                                     , A.GridDistortion()
                                                     , A.CLAHE(clip_limit=2)
                                                     , A.RandomGamma()
                                                     , A.RandomBrightnessContrast()
                                                  ], p=1
                                                 )
                                            , A.Resize(self.cfg.mode.crop.width, self.cfg.mode.crop.height)
                                            , ToTensorV2()])
            transformed = train_transform(image=img_raw, mask=img_mask)
            transformed_img = transformed['image']
            # Albumentations returns masks of the form HxWxC. We permute to make channels first
            transformed_mask = transformed['mask'].long()
            # Tensors are to be converted to float tensors as cv2 is giving byte tensor
            transformed_img = transformed_img.type(torch.FloatTensor)
            return transformed_img, transformed_mask
        elif self.aug_type == 'val':
            val_transform = A.Compose(
                [A.RandomBrightnessContrast(p=0.3), A.Resize(self.cfg.mode.crop.width, self.cfg.mode.crop.height),
                 ToTensorV2()])
            transformed = val_transform(image=img_raw, mask=img_mask)
            transformed_img = transformed['image']
            transformed_mask = transformed['mask'].long()
            # Tensors are to be converted to float tensors as cv2 is giving byte tensor
            transformed_img = transformed_img.type(torch.FloatTensor)
            return transformed_img, transformed_mask
        else:
            test_transform = A.Compose(
                [A.RandomBrightnessContrast(p=0.3), A.Resize(self.cfg.mode.crop.width, self.cfg.mode.crop.height),
                 ToTensorV2()])
            transformed = test_transform(image=img_raw, mask=img_mask)
            transformed_img = transformed['image']
            transformed_mask = transformed['mask'].long()
            # Tensors are to be converted to float tensors as cv2 is giving byte tensor
            transformed_img = transformed_img.type(torch.FloatTensor)
            return transformed_img, transformed_mask
