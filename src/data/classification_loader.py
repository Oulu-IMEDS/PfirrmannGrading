import albumentations as A
import cv2
import logging
import numpy as np
import random
import torch
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class ClassificationLoader(Dataset):
    def __init__(self, cfg, data_slice, aug_type='train'):
        # Hydra Configuration
        self.cfg = cfg
        # Data slice of overall dataset
        self.data_slice = data_slice
        # Size of the Data Slice
        self.dataset_size = data_slice.shape[0]
        # Apply transformations based on train or val set
        self.aug_type = aug_type
        # Class names
        self.class_id = {'2': 0, '3': 1, '4': 2, '5': 3}

    def __len__(self):
        return self.dataset_size

    def get_weighted_slices(self, slice_volume):
        weight_vector = []
        midpoint = slice_volume // 2

        # if we are processing only one slice, return the mid-sagittal slice
        if self.cfg.mode.classification.num_slices == 1:
            weight_vector = np.zeros(slice_volume)
            weight_vector[midpoint] = 1
            return weight_vector

        # Create weights which are monotonically increasing till midpoint and decrease further
        for i in range(midpoint):
            weight_vector.append(int(i + 0.5))
        for i in np.arange(midpoint, slice_volume):
            weight_vector.append(int(slice_volume - 2 * i * 0.5))

        return weight_vector

    def __getitem__(self, index):
        # Get the filename of the image
        # get n images from the folder based on the configuration. If we do not have enough, repeat the image
        data_dir = self.data_slice.iloc[index]['dir_name']
        image_list = sorted(list(Path(self.cfg.mode.classification.dir, data_dir).glob("*.png")))

        # The dataloader should give same amount of images.
        # So if there are no sufficient images, we use existing images
        # Give more weightage to the central slices of the MRI

        if len(image_list) < self.cfg.mode.classification.num_slices:
            deficit = self.cfg.mode.classification.num_slices - len(image_list)
            # Append the last image with the deficit through random selection
            weight_vector = self.get_weighted_slices(len(image_list))
            image_list += sorted(random.choices(image_list, weights=weight_vector, k=deficit))
        elif len(image_list) >= self.cfg.mode.classification.num_slices:
            weight_vector = self.get_weighted_slices(len(image_list))
            image_list = sorted(
                random.choices(image_list, weights=weight_vector, k=self.cfg.mode.classification.num_slices))

        transformed_images = []
        labels = []
        for image_path in image_list:
            # We have only Pfirrmann Grades 2,3,4,5 in the dataset. Map these classes to ids starting from 0 index
            pfirrmann_grade = self.class_id[image_path.name[-5:][0]]
            img_raw = cv2.imread(str(image_path))
            # Converting the image to single channel grayscale
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
            # Collapse the single channel by applying squeeze
            transformed_img = self.apply_img_transforms(img_raw).squeeze()
            transformed_images.append(transformed_img)
        # Label for all the images are the same. So we append after the loop
        labels.append(pfirrmann_grade)

        # Stack the images in depth dimension
        transformed_images = torch.Tensor(np.stack(transformed_images))
        labels = torch.Tensor(labels).long().squeeze()

        return {"images": transformed_images, "labels": labels}

    # Apply Image Transformations using Albumentations
    def apply_img_transforms(self, img_raw):
        if self.aug_type == 'train':
            train_transform = A.Compose(
                [A.OneOf([A.RandomGamma(),
                          A.RandomBrightnessContrast(brightness_limit=(0, 0.2), contrast_limit=(0, 0.1)),
                          ], p=0.5
                         ),
                 # A.Resize(self.cfg.mode.classification.img_width, self.cfg.mode.classification.img_height),
                 A.Normalize(mean=0.235, std=0.134, always_apply=True, p=1.0),
                 ToTensorV2()])
            transformed = train_transform(image=img_raw)
            transformed_img = transformed['image']
            transformed_img = transformed_img.type(torch.FloatTensor)
            return transformed_img
        elif self.aug_type == 'val':
            val_transform = A.Compose(
                [A.OneOf([A.RandomGamma(),
                          A.RandomBrightnessContrast(brightness_limit=(0, 0.2), contrast_limit=(0, 0.1)),
                          ], p=0.5
                         ),
                 # A.Resize(self.cfg.mode.classification.img_width, self.cfg.mode.classification.img_height),
                 A.Normalize(mean=0.235, std=0.134, always_apply=True, p=1.0),
                 ToTensorV2()])
            transformed = val_transform(image=img_raw)
            transformed_img = transformed['image']
            transformed_img = transformed_img.type(torch.FloatTensor)
            return transformed_img
        else:
            test_transform = A.Compose(
                [  # A.Resize(self.cfg.mode.classification.img_width, self.cfg.mode.classification.img_height),
                    A.Normalize(mean=0.235, std=0.134, always_apply=True, p=1.0),
                    ToTensorV2()])
            transformed = test_transform(image=img_raw)
            transformed_img = transformed['image']
            # Tensors are to be converted to float tensors as cv2 is giving byte tensor
            transformed_img = transformed_img.type(torch.FloatTensor)
            return transformed_img
