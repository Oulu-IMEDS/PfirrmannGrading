import gc
import numpy as np
import pathlib
import torch
from pathlib import Path, PurePath
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from src.data.segmentation_loader import SegmentationLoader
from src.model.model_architectures import build_model
from src.utils.segmentation_utils import calculate_iou, calculate_confusion_matrix_from_arrays, \
    visualize_random_img_masks


def execute_training(cfg, epoch, train_loader, val_loader, segmentation_model, criterion, optimizer, logger, writer):
    train_loss = 0
    iou_cum = []
    num_batches = 0
    with tqdm(train_loader, unit="batch") as tepoch:
        segmentation_model.train()
        for i_batch, batch in enumerate(tepoch):
            tepoch.set_description(f"Epoch:{epoch} Training:")
            num_batches += 1
            images = batch['transformed_raw'].to(cfg.device)
            masks = batch['transformed_mask'].to(cfg.device)

            # Don't squeeze the dimensions if there is only one dataset point
            if cfg.training.loss_function['_target_'] == 'segmentation_models_pytorch.losses.FocalLoss':
                if len(masks.shape) == 4 and masks.shape[0] != 1:
                    masks = masks.squeeze()
                else:
                    masks = masks.squeeze(0)

            with torch.set_grad_enabled(mode=True):
                outputs = segmentation_model(images)
                optimizer.zero_grad()
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            tepoch.set_postfix(train_loss=train_loss / num_batches)

    with tqdm(val_loader, unit="batch") as vepoch:
        with torch.no_grad():
            segmentation_model.eval()
            vepoch.set_description(f"\t Validation: ")
            for i_batch, batch in enumerate(vepoch):
                images = batch['transformed_raw'].to(cfg.device)
                masks = batch['transformed_mask'].to(cfg.device)

                outputs = segmentation_model(images)
                preds = outputs.argmax(axis=1)
                preds = preds.cpu().detach().float().numpy()
                masks = masks.cpu().detach().float().numpy()

                # Calculate the confusion matrix and IOC for the epoch
                for batch_el_id in range(preds.shape[0]):
                    confusion_matrix = calculate_confusion_matrix_from_arrays(preds[batch_el_id, :, :],
                                                                              masks[batch_el_id, :, :],
                                                                              cfg.mode.input.n_classes)
                    iou_res = calculate_iou(confusion_matrix)
                    iou_cum.append(np.array(iou_res))
                vepoch.set_postfix(mean_iou=np.mean(iou_cum))

    writer.add_scalars('Loss',
                       {'train_loss': train_loss / num_batches}, epoch)
    writer.add_scalars('IOU', {'IOU': np.mean(iou_cum)}, epoch)
    return np.mean(iou_cum)


def save_model_checkpoint(cfg, segmentation_model, epoch, epoch_iou):
    model_checkpoint = {"epoch": epoch, "model_state": segmentation_model.module.state_dict(), "epoch_iou": epoch_iou}
    fname = f"SpineSeg_Epoch{epoch}_EpochIOU{epoch_iou:.4f}.pth"
    checkpoint_file = PurePath.joinpath(Path.cwd(), "results", "models", fname)
    prev_model = list(
        pathlib.Path(PurePath.joinpath(Path.cwd(), "results", "models")).glob(
            "SpineSeg_*"))
    if len(prev_model) >= 1:
        Path.unlink(prev_model[0], missing_ok=False)
    torch.save(model_checkpoint, checkpoint_file)
    return checkpoint_file


def start_training(cfg, segmentation_df, logger):
    # Build the model
    segmentation_model, criterion, optimizer, scheduler = build_model(cfg, logger)
    segmentation_model = segmentation_model.to(cfg.device)
    logger.info(f"Using Model Architecture:{cfg.segmentation_architecture['_target_']}")

    logdir = Path(PurePath.joinpath(Path.cwd(), "results", "plots"))
    writer = SummaryWriter(logdir)

    if cfg.device == 'cuda':
        segmentation_model = DataParallel(segmentation_model)

    split_mode = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=cfg.random_seed)
    data_split = split_mode.split(segmentation_df, groups=segmentation_df['patient_id'])
    train_index, val_index = next(data_split)

    train_df = segmentation_df.loc[train_index]
    val_df = segmentation_df.loc[val_index]

    logger.info(f"Starting Training with Train DF Shape:{train_df.shape} Val DF Shape:{val_df.shape}")

    return

    # Create segmentation dataset with Augmentations
    train_ds = SegmentationLoader(cfg, train_df, 'train')
    val_ds = SegmentationLoader(cfg, val_df, 'val')

    # Create the train and val loaders
    train_loader = DataLoader(dataset=train_ds, batch_size=cfg.training.dataloader.batch_size, shuffle=True,
                              num_workers=cfg.training.dataloader.num_workers)
    val_loader = DataLoader(dataset=val_ds, batch_size=cfg.training.dataloader.batch_size,
                            num_workers=cfg.training.dataloader.num_workers)

    # Visualize random image and mask
    visualize_random_img_masks(train_loader, writer, logger)

    prev_iou = 0
    for epoch in range(cfg.training.epochs):
        epoch_iou = execute_training(cfg, epoch, train_loader, val_loader, segmentation_model, criterion,
                                     optimizer, logger, writer)
        # Learning Rate Decay
        scheduler.step()

        # Start saving checkpoints after the configured epochs passed
        if epoch > cfg.training.checkpoint.epochs_to_pass and prev_iou < epoch_iou:
            prev_iou = epoch_iou
            checkpoint_file = save_model_checkpoint(cfg, segmentation_model, epoch, epoch_iou)
            # checkpoint_message = "Checkpoint {} saved".format(checkpoint_file)
            # logger.info(checkpoint_message)

    writer.flush()
    writer.close()
    gc.collect()
