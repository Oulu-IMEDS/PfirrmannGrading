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

from src.data.multimodal_classification_loader import MultiModalLoader
from src.model.model_architectures import build_model
from src.utils.multimodal_utils import visualize_random_imgs
from src.training.focal_loss import FocalLoss


def execute_training(cfg, epoch, train_loader, val_loader, multimodal_classifier, criterion, optimizer, scheduler,
                     logger, y_true_epoch, y_pred_epoch, writer):
    train_loss = 0
    num_tr_batches = 0
    num_val_batches = 0
    total = 0
    correct_preds = 0
    with tqdm(train_loader, unit="batch") as tepoch:
        multimodal_classifier.train()
        for i_batch, batch in enumerate(tepoch):
            tepoch.set_description(f"Epoch:{epoch} Training:")
            num_tr_batches += 1

            inputs = batch
            labels = batch['labels'].to(cfg.device)

            with torch.set_grad_enabled(mode=True):
                multimodal_classifier = multimodal_classifier.to(cfg.device)
                outputs = multimodal_classifier(inputs)
                optimizer.zero_grad()

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
            multimodal_classifier.eval()
            vepoch.set_description(f"\t Validation: ")
            for i_batch, batch in enumerate(vepoch):
                num_val_batches += 1
                inputs = batch
                labels = batch['labels'].to(cfg.device)

                outputs = multimodal_classifier(inputs)
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


def save_model_checkpoint(cfg, multimodal_classifier, epoch, epoch_acc):
    model_checkpoint = {"epoch": epoch, "model_state": multimodal_classifier.state_dict(), "val_acc": epoch_acc}
    fname = f"SpineMMCls_Epoch{epoch}_Bal_Acc{epoch_acc:.4f}.pth"
    checkpoint_file = PurePath.joinpath(Path.cwd(), "results", "models", fname)
    prev_model = list(
        Path(PurePath.joinpath(Path.cwd(), "results", "models")).glob("SpineMMCls_*"))
    if len(prev_model) >= 1:
        Path.unlink(prev_model[0], missing_ok=False)
    torch.save(model_checkpoint, checkpoint_file)
    return checkpoint_file


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

    return criterion


def start_multimodal_classification(cfg, multimodal_df, logger):
    # Build the model
    multimodal_classifier, criterion, optimizer, scheduler = build_model(cfg, logger)
    logger.info(f"Using Model Architecture")

    logdir = Path(PurePath.joinpath(Path.cwd(), "results", "plots"))
    writer = SummaryWriter(logdir)

    if cfg.device == 'cuda':
        multimodal_model = DataParallel(multimodal_classifier)

    split_mode = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=cfg.random_seed)
    data_split = split_mode.split(multimodal_df, groups=multimodal_df['patient_id'])
    train_index, val_index = next(data_split)

    train_df = multimodal_df.loc[train_index]
    val_df = multimodal_df.loc[val_index]

    # Building weighted cross entropy loss function
    target_col = cfg.mode.multimodal.target_column
    class_weights = compute_class_weight(class_weight='balanced', classes=train_df[target_col].unique(),
                                         y=train_df[target_col])
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(cfg.device)

    criterion = get_loss_function(cfg, logger, class_weights)

    # Create Multimodal dataset with Augmentations
    train_ds = MultiModalLoader(cfg, train_df, 'train')
    val_ds = MultiModalLoader(cfg, val_df, 'val')

    train_sampler = get_weighted_sampler(cfg, train_df)
    val_sampler = get_weighted_sampler(cfg, val_df)

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
    y_true_epoch = []
    y_pred_epoch = []
    print(f"Starting Training with Train Set:{train_df.shape} Validation Set:{val_df.shape}")
    for epoch in range(cfg.training.epochs):
        epoch_acc, epoch_bal_acc = execute_training(cfg, epoch, train_loader, val_loader, multimodal_classifier,
                                                    criterion,
                                                    optimizer, scheduler, logger, y_true_epoch, y_pred_epoch, writer)

        # Scheduler Step
        if cfg.training.scheduler.type == 'step':
            scheduler.step()

        # Start saving checkpoints after configured epochs passed
        measure = epoch_bal_acc
        if epoch >= cfg.training.checkpoint.epochs_to_pass and prev_acc < measure:
            prev_acc = measure
            checkpoint_file = save_model_checkpoint(cfg, multimodal_classifier, epoch, measure)
            # checkpoint_message = "Checkpoint {} saved".format(checkpoint_file)
            # logger.info(checkpoint_message)

    writer.flush()
    writer.close()
    gc.collect()
