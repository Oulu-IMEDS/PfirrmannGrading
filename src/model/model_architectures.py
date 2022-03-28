import hydra
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

def build_model(cfg,logger):
    if cfg.architecture['_target_'] == 'segmentation_models_pytorch.Unet':
        logger.info("Using UNET architecture")
        segmentation_model = hydra.utils.instantiate(cfg.architecture)
    elif cfg.architecture['_target_'] == 'segmentation_models_pytorch.FPN':
        logger.info("Using FPN architecture")
        segmentation_model = hydra.utils.instantiate(cfg.architecture)

    criterion=hydra.utils.instantiate(cfg.loss_function)
    optimizer = Adam(segmentation_model.parameters(),weight_decay=cfg.model.optimizer.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=cfg.model.scheduler.factor, patience=cfg.model.scheduler.patience)

    return segmentation_model, criterion, optimizer, scheduler
