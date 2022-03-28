import os
import hydra
import gc
import logging
import numpy as np
import torch
from omegaconf import DictConfig
from src.data.segmentation_loader import SegmentationLoader
from src.data.segmentation_metadata import build_segmentation_metadata
from src.utils.utils import visualize_random_img_masks
from src.utils.utils import visualize_segmentation
from src.model.model_architectures import build_model
from src.model.train import start_training
from torch.utils.data import DataLoader, SequentialSampler

# A logger for this file
logger = logging.getLogger(__name__)

def set_random_seed(cfg):
    torch.cuda.empty_cache()
    gc.collect()
    np.random.seed(cfg.random_seed.seed)


@hydra.main(config_path="configs", config_name="configs")
def start_segmentation_pipeline(cfg: DictConfig) -> None:
    set_random_seed(cfg)

    if cfg.device=='cuda':
        logger.info("Starting the pipeline with device: "+cfg.device)
    else:
        logger.warning("Starting the pipeline with device: " + cfg.device)

    # Build Segmentation Metadata from Images and Masks directories
    train_df, val_df, test_df = build_segmentation_metadata(cfg,logger)

    # Create segmentation dataset with Augmentations
    train_ds = SegmentationLoader(cfg, train_df, 'train')
    val_ds = SegmentationLoader(cfg, val_df, 'val')
    val_sampler = SequentialSampler(val_ds)

    # Visualize random image and mask
    # visualize_random_img_masks(train_ds,logger)

    # Create the train and val loaders
    train_loader = DataLoader(dataset=train_ds, batch_size=cfg.data.dataloader.batch_size, shuffle=True,
                              num_workers=cfg.data.dataloader.num_workers)
    val_loader = DataLoader(dataset=val_ds, batch_size=cfg.data.dataloader.batch_size,
                            num_workers=cfg.data.dataloader.num_workers, sampler=val_sampler)

    # Build the model
    model, criterion, optimizer, scheduler = build_model(cfg,logger)

    # Start the model training
    start_training(cfg,train_loader, val_loader, model, criterion, optimizer, scheduler,logger)

    #Visualize random segmentation
    #visualize_segmentation(cfg)


if __name__ == '__main__':
    start_segmentation_pipeline()
