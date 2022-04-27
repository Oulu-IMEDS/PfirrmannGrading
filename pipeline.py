import gc
import hydra
import imgaug
import logging
import numpy as np
import random
import torch
import cv2
from omegaconf import DictConfig
from pathlib import Path, PurePath

from src.data.classification_metadata import build_classification_metadata
from src.data.segmentation_metadata import build_segmentation_metadata
from src.training.classify import start_classification
from src.training.segment import start_training
from src.utils.classification_utils import score_avg_classification, split_train_test
from src.utils.segmentation_utils import visualize_segmentation, generate_spine_map, generate_mri_labels

# logger for generating log file
logger = logging.getLogger(__name__)


def create_sub_dir(cfg):
    sub_dirs = ['models', 'plots', 'pickles']
    for sub_dir in sub_dirs:
        dir_path = Path(PurePath.joinpath(Path.cwd(), "results", sub_dir))
        dir_path.mkdir(parents=True, exist_ok=True)


def set_random_seed(cfg):
    torch.cuda.empty_cache()
    gc.collect()
    np.random.seed(cfg.random_seed)
    torch.cuda.manual_seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    imgaug.random.seed(cfg.random_seed)
    random.seed(cfg.random_seed)
    cv2.setRNGSeed(cfg.random_seed)


def start_segmentation_pipeline(cfg: DictConfig) -> None:
    if cfg.mode.segmentation.run_mode == 'train':
        if cfg.device == 'cuda':
            logger.info("Starting the segmentation pipeline with device: " + cfg.device)
        else:
            logger.warning("Starting the segmentation pipeline with device: " + cfg.device)

        # Build Segmentation Metadata from Images and Masks directories
        segmentation_df = build_segmentation_metadata(cfg, logger)
        start_training(cfg, segmentation_df, logger)
    elif cfg.mode.segmentation.run_mode == 'view':
        # Visualize random segmentation
        input = [fname for fname in list(PurePath.joinpath(Path(cfg.mode.test.dir)).glob("*.png"))]
        index = np.random.randint(0, len(input))
        logger.info("Visualizing the file:{}".format(input[index]))
        visualize_segmentation(cfg, str(input[index]), logger)
    elif cfg.mode.segmentation.run_mode == 'spine-map':
        generate_spine_map(cfg, logger)
    elif cfg.mode.segmentation.run_mode == 'classify-labels':
        generate_mri_labels(cfg, logger)


def start_classification_pipeline(cfg: DictConfig) -> None:
    if cfg.mode.classification.run_mode == 'train':
        if cfg.device == 'cuda':
            logger.info("Starting the classification pipeline with device: " + cfg.device)
        else:
            logger.warning("Starting the classification pipeline with device: " + cfg.device)

        # Build Segmentation Metadata from Images and Masks directories
        cdf = build_classification_metadata(cfg, logger)

        # split_train_test method will split into train_df (80%) and 4 tests_dfs
        classification_df, test_dfs = split_train_test(cfg, cdf)
        start_classification(cfg, classification_df, logger)

        avg_model_score = []
        for idx, test_df in enumerate(test_dfs):
            avg_model_score.append(score_avg_classification(cfg, idx, test_df, logger))
        logger.info(f"Model Score:{round(np.mean(avg_model_score), 3)}, std:{round(np.std(avg_model_score), 3)} ")


@hydra.main(config_path="configs", config_name="configs")
def start_pipeline(cfg: DictConfig) -> None:
    pipeline = list(cfg.mode.keys())[0]
    if pipeline == 'segmentation':
        set_random_seed(cfg)
        create_sub_dir(cfg)
        start_segmentation_pipeline(cfg)
    elif pipeline == 'classification':
        set_random_seed(cfg)
        create_sub_dir(cfg)
        start_classification_pipeline(cfg)


if __name__ == '__main__':
    start_pipeline()
    # model = SpineNet()
    # output = model(torch.randn(1, 9, 112, 224))
    # print(output.shape)
