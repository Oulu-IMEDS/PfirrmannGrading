import gc
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path, PurePath
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import compute_class_weight
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.classification_loader import ClassificationLoader
from src.model.model_architectures import build_model
from src.utils.classification_utils import visualize_random_imgs
from src.training.focal_loss import FocalLoss
from src.training.balanced_loss import ClassBalancedloss


def execute_training(cfg, epoch, train_loader, val_loader, classification_model, criterion, optimizer, scheduler,
                     logger, y_true_epoch, y_pred_epoch, no_class_samples, writer):
    train_loss = 0
    num_tr_batches = 0
    num_val_batches = 0
    total = 0
    correct_preds = 0
    with tqdm(train_loader, unit="batch") as tepoch:
        classification_model.train()
        for i_batch, batch in enumerate(tepoch):
            tepoch.set_description(f"Epoch:{epoch} Training:")
            num_tr_batches += 1
            if cfg.classification_architecture['_target_'] == 'spinenetv2':
                images = batch['images'].to(cfg.device).unsqueeze(1)
            else:
                images = batch['images'].to(cfg.device)
            labels = batch['labels'].to(cfg.device)
            with torch.set_grad_enabled(mode=True):
                classification_model = classification_model.to(cfg.device)
                outputs = classification_model(images)
                # print(outputs)
                optimizer.zero_grad()
                if cfg.training.loss_function == 'CB':
                    loss = ClassBalancedloss(cfg, outputs, labels, no_class_samples)
                else:
                    loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # Scheduler should step per batch for Cyclic LR
                if cfg.training.scheduler.type == 'cyclic':
                    scheduler.step()

            train_loss += loss.item()
            _, index = torch.max(outputs, 1)

            total += labels.size(0)
            correct_preds += (index == labels).sum().item()
            train_acc = 100 * (correct_preds / total)
            tepoch.set_postfix(train_acc=train_acc, train_loss=train_loss / num_tr_batches)

    with tqdm(val_loader, unit="batch") as vepoch:
        with torch.no_grad():
            classification_model.eval()
            vepoch.set_description(f"\t Validation: ")
            for i_batch, batch in enumerate(vepoch):
                num_val_batches += 1
                if cfg.classification_architecture['_target_'] == 'spinenetv2':
                    images = batch['images'].to(cfg.device).unsqueeze(1)
                else:
                    images = batch['images'].to(cfg.device)

                labels = batch['labels'].to(cfg.device)
                outputs = classification_model(images)

                if cfg.training.loss_function == 'CB':
                    val_loss = ClassBalancedloss(cfg, outputs, labels, no_class_samples)
                else:
                    val_loss = criterion(outputs, labels)
                _, index = torch.max(outputs, 1)

                total += labels.size(0)
                correct_preds += (index == labels).sum().item()
                val_loss += val_loss.item()
                val_acc = 100 * (correct_preds / total)

                y_true_epoch.append(labels.detach().cpu().numpy().squeeze())
                y_pred_epoch.append(index.detach().cpu().numpy().squeeze())
                bal_acc = balanced_accuracy_score(np.concatenate(y_true_epoch), np.concatenate(y_pred_epoch)) * 100
                vepoch.set_postfix(val_loss=val_loss.item() / num_val_batches, val_acc=val_acc, bal_acc=bal_acc)
    writer.add_scalars('Loss',
                       {'train_loss': train_loss / num_tr_batches, 'val_loss': val_loss / num_val_batches}, epoch)
    writer.add_scalars('Accuracy', {'train_acc': train_acc, 'val_acc': val_acc, 'bal_acc': bal_acc}, epoch)
    bal_acc = balanced_accuracy_score(np.concatenate(y_true_epoch), np.concatenate(y_pred_epoch)) * 100

    return val_acc, bal_acc


def save_model_checkpoint(cfg, classification_model, epoch, epoch_acc):
    model_checkpoint = {"epoch": epoch, "model_state": classification_model.module.state_dict(), "val_acc": epoch_acc}
    fname = f"SpineCls_Epoch{epoch}_Bal_Acc{epoch_acc:.4f}.pth"
    checkpoint_file = PurePath.joinpath(Path.cwd(), "results", "models", fname)
    prev_model = list(
        Path(PurePath.joinpath(Path.cwd(), "results", "models")).glob("SpineCls_*"))
    if len(prev_model) >= 1:
        Path.unlink(prev_model[0], missing_ok=False)
    torch.save(model_checkpoint, checkpoint_file)
    return checkpoint_file


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))


def get_no_class_samples(cfg, df):
    # Create a weighted train sampler
    class_idx = {'2': 0, '3': 1, '4': 2, '5': 3}
    y_train_indices = df.pfirrmann_grade.index
    y_train = [class_idx[str(df.pfirrmann_grade[i])] for i in y_train_indices]
    class_sample_count = np.unique(y_train, return_counts=True)[1]
    return class_sample_count


def get_weighted_sampler(cfg, df):
    # Create a weighted train sampler
    class_idx = {'2': 0, '3': 1, '4': 2, '5': 3}
    y_train_indices = df.pfirrmann_grade.index
    y_train = [class_idx[str(df.pfirrmann_grade[i])] for i in y_train_indices]

    class_sample_count = np.unique(y_train, return_counts=True)[1]
    weight = 1. / class_sample_count
    samples_weight = weight[y_train]
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler


def get_loss_function(cfg, logger, class_weights):
    if cfg.training.loss_function == 'CE':
        logger.info('Using Cross Entropy Loss')
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)
    elif cfg.training.loss_function == 'WCE':
        logger.info('Using Weighted Cross Entropy Loss')
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.training.label_smoothing)
    elif cfg.training.loss_function == 'FL':
        logger.info('Using Focal Loss')
        criterion = FocalLoss(label_smoothing=cfg.training.label_smoothing)
    elif cfg.training.loss_function == 'FLWCE':
        logger.info('Using Focal Loss with Weighted Cross Entropy')
        criterion = FocalLoss(weight=class_weights, label_smoothing=cfg.training.label_smoothing)
    elif cfg.training.loss_function == 'CB':
        logger.info('Using Class Balanced Loss')
        criterion = None
    return criterion


def start_classification(cfg, classification_df, logger):
    # Build the model
    classification_model, criterion, optimizer, scheduler = build_model(cfg, logger)
    logger.info(f"Using Model Architecture:{cfg.classification_architecture['_target_']}")

    if cfg.classification_architecture['_target_'] == 'spinenetv1':
        classification_model.apply(init_weights)

    logdir = Path(PurePath.joinpath(Path.cwd(), "results", "plots"))
    writer = SummaryWriter(logdir)

    if cfg.device == 'cuda':
        classification_model = DataParallel(classification_model)

    split_mode = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=cfg.random_seed)
    data_split = split_mode.split(classification_df, groups=classification_df['patient_id'])
    train_index, val_index = next(data_split)

    train_df = classification_df.loc[train_index]
    val_df = classification_df.loc[val_index]

    logger.info(f"Train Size:{train_df.shape[0]}, Val Size:{val_df.shape[0]}")

    # Building weighted cross entropy loss function
    class_weights = compute_class_weight(class_weight='balanced', classes=train_df.pfirrmann_grade.unique(),
                                         y=train_df.pfirrmann_grade)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(cfg.device)

    # Create Classification dataset with Augmentations
    train_ds = ClassificationLoader(cfg, train_df, 'train')
    val_ds = ClassificationLoader(cfg, val_df, 'val')

    train_sampler = get_weighted_sampler(cfg, train_df)
    val_sampler = get_weighted_sampler(cfg, val_df)

    no_class_samples = get_no_class_samples(cfg, classification_df)
    criterion = get_loss_function(cfg, logger, class_weights)

    # Create the train and val loaders
    if cfg.training.sampler == 'weighted':
        logger.info(f"Using Weighted Sampler")
        train_loader = DataLoader(dataset=train_ds, batch_size=cfg.training.dataloader.batch_size,
                                  num_workers=cfg.training.dataloader.num_workers, sampler=train_sampler)
        val_loader = DataLoader(dataset=val_ds, batch_size=cfg.training.dataloader.batch_size,
                                num_workers=cfg.training.dataloader.num_workers, sampler=val_sampler)
    else:
        logger.info(f"Using Default Sampler")
        train_loader = DataLoader(dataset=train_ds, batch_size=cfg.training.dataloader.batch_size,
                                  num_workers=cfg.training.dataloader.num_workers)
        val_loader = DataLoader(dataset=val_ds, batch_size=cfg.training.dataloader.batch_size,
                                num_workers=cfg.training.dataloader.num_workers)

    # Visualize random image and mask
    visualize_random_imgs(train_loader, writer, logger)

    prev_acc = 0
    no_improvement_cnt = 0
    y_true_epoch = []
    y_pred_epoch = []
    logger.info(f"Starting Training with Train Set:{train_df.shape} Validation Set:{val_df.shape}")
    for epoch in range(cfg.training.epochs):
        epoch_acc, epoch_bal_acc = execute_training(cfg, epoch, train_loader, val_loader, classification_model,
                                                    criterion,
                                                    optimizer, scheduler, logger, y_true_epoch, y_pred_epoch,
                                                    no_class_samples, writer)

        # Scheduler Step
        if cfg.classification_architecture['_target_'] == 'spinenetv1' or cfg.classification_architecture['_target_'] == 'spinenet_miccai':
            scheduler.step(epoch_bal_acc)
        elif cfg.training.scheduler.type == 'step' and cfg.classification_architecture['_target_'] != 'spinenetv2':
            scheduler.step()

        # Start saving checkpoints after configured epochs passed
        measure = epoch_bal_acc
        if epoch >= cfg.training.checkpoint.epochs_to_pass:
            if np.round(prev_acc, 3) < np.round(measure, 3):
                prev_acc = measure
                checkpoint_file = save_model_checkpoint(cfg, classification_model, epoch, measure)
            # Implementing the 10 epoch balanced accuracy for spinenet
            elif cfg.classification_architecture['_target_'] == 'spinenetv2':
                no_improvement_cnt += 1
                # There is no improvement in balanced accuracy for last 10 epochs
                if no_improvement_cnt == 10:
                    logger.info("No improvement for past 10 epochs.. Exiting")
                    break
            # checkpoint_message = "Checkpoint {} saved".format(checkpoint_file)
            # logger.info(checkpoint_message)

    writer.flush()
    writer.close()
    gc.collect()
