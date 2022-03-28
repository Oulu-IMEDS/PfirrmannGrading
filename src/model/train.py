import gc
import os
import torch
import numpy as np
from src.utils.utils import calculate_iou, calculate_confusion_matrix_from_arrays


def execute_training(cfg, loader, mode, segmentation_model, criterion, optimizer, logger):
    train_loss = 0
    iou_cum = []
    for i_batch, batch in enumerate(loader):
        images = batch['transformed_raw'].to(cfg.device)
        masks = batch['transformed_mask'].to(cfg.device)

        # Don't squeeze the dimensions if there is only one dataset point
        if len(masks.shape) == 3:
            masks = masks.squeeze()

        segmentation_model = segmentation_model.to(cfg.device)
        outputs = segmentation_model(images)

        if mode == 'training':
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
        elif mode == 'validation':
            preds = outputs.argmax(axis=1)
            preds = preds.cpu().detach().float().numpy()
            masks = masks.cpu().detach().float().numpy()

            # Calculate the confusion matrix and IOC for the epoch
            for batch_el_id in range(preds.shape[0]):
                confusion_matrix = calculate_confusion_matrix_from_arrays(preds[batch_el_id, :, :],
                                                                          masks[batch_el_id, :, :], cfg.model.n_classes)
                iou_res = calculate_iou(confusion_matrix)
                iou_cum.append(np.array(iou_res))

    if mode == 'training':
        return train_loss
    else:
        return np.mean(iou_cum)


def save_model_checkpoint(segmentation_model, epoch, epoch_iou):
    model_checkpoint = {"epoch": epoch, "model_state": segmentation_model.state_dict(), "epoch_iou": epoch_iou}
    fname = "SpineSeg_Epoch{}_EpochIOU{}.pth".format(epoch, np.round(epoch_iou,4))
    checkpoint_file = os.path.join(os.getcwd(), "results", "models", fname)
    torch.save(model_checkpoint, checkpoint_file)
    return checkpoint_file


def start_training(cfg, train_loader, val_loader, segmentation_model, criterion, optimizer, scheduler, logger):
    prev_iou = 0
    for epoch in range(cfg.model.epochs):
        epoch_train_loss = execute_training(cfg, train_loader, 'training', segmentation_model, criterion, optimizer,
                                            logger)
        epoch_iou = execute_training(cfg, val_loader, 'validation', segmentation_model, criterion, optimizer, logger)

        # Learning Rate Decay
        scheduler.step(epoch_iou)

        # Start saving checkpoints only after 5 epochs
        if prev_iou < epoch_iou and epoch > 5:
            prev_iou = epoch_iou
            checkpoint_file = save_model_checkpoint(segmentation_model, epoch, epoch_iou)
            checkpoint_message = "Checkpoint {} saved".format(checkpoint_file)
        else:
            checkpoint_message = ""
        logger.info(
            "Epoch:{}/{}, Train Loss:{}, Mean IOU:{}, {}".format(epoch, cfg.model.epochs, epoch_train_loss, epoch_iou,
                                                                 checkpoint_message))
        gc.collect()
