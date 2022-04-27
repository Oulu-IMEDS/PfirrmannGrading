import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from pathlib import Path, PurePath
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GroupShuffleSplit
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from src.data.classification_loader import ClassificationLoader
from src.model.model_architectures import build_model


def split_test(cfg, df):
    test_dfs = []
    split_mode = GroupShuffleSplit(test_size=0.5, n_splits=1,
                                   random_state=cfg.random_seed)
    data_split = split_mode.split(df, groups=df['patient_id'])
    subset1_ind, subset2_ind = next(data_split)
    subset_test1_df = df.iloc[subset1_ind]
    subset_test2_df = df.iloc[subset2_ind]

    data_split = split_mode.split(subset_test1_df, groups=subset_test1_df['patient_id'])
    test1_ind, test2_ind = next(data_split)
    test1_df = subset_test1_df.iloc[test1_ind]
    test_dfs.append(test1_df)
    test2_df = subset_test1_df.iloc[test2_ind]
    test_dfs.append(test2_df)

    data_split = split_mode.split(subset_test2_df, groups=subset_test2_df['patient_id'])
    test3_ind, test4_ind = next(data_split)
    test3_df = subset_test2_df.iloc[test3_ind]
    test_dfs.append(test3_df)
    test4_df = subset_test2_df.iloc[test4_ind]
    test_dfs.append(test4_df)

    return test_dfs


def split_train_test(cfg, df):
    # Split the main df into train,test splits
    split_mode = GroupShuffleSplit(test_size=cfg.training.train_test.test_perc, n_splits=1,
                                   random_state=cfg.random_seed)
    data_split = split_mode.split(df, groups=df['patient_id'])
    train_inds, test_inds = next(data_split)

    classification_df = df.iloc[train_inds]
    classification_df = classification_df.reset_index()
    test_dfs = split_test(cfg, df.iloc[test_inds])
    return classification_df, test_dfs


def visualize_random_imgs(image_batch, writer, logger):
    # print first 16 images of the train loader
    plt.figure(figsize=(20, 20))
    idx, imgs = next(enumerate(image_batch))
    for index in range(16):
        image_1 = imgs['images'][index][0].unsqueeze(0)  # .long().numpy()
        label = imgs['labels'][index]
        # Inverse Normalize the image before displaying
        inv_normalize = transforms.Normalize(
            mean=[-m / s for m, s in zip([0.235], [0.134])],
            std=[1 / s for s in [0.134]]
        )

        inv_tensor = inv_normalize(image_1)

        plt.subplot(4, 4, index + 1)
        plt.title(f"Pfirrmann Grade:{label + 2}")
        plt.imshow(inv_tensor.squeeze(0), cmap='gray')
    plt.savefig('train_images.png', dpi='figure')
    img_raw = cv2.imread(str('train_images.png'))
    images_train = torch.tensor(cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY))
    writer.add_image(f"Random Images from Train Set", images_train, dataformats='HW')
    plt.close()


def save_cm_kfold(cfg, fold, y_true, y_pred, logger):
    # Save confusion matrix results per fold
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # Save the labels and predictions for later use
    y_true_file = f"y_true_{fold}.npy"
    y_pred_file = f"y_pred_{fold}.npy"

    open_file = open(PurePath.joinpath(Path.cwd(), "results", "pickles", y_true_file), "wb")
    np.save(open_file, y_true)
    open_file.close()

    open_file = open(PurePath.joinpath(Path.cwd(), "results", "pickles", y_pred_file), "wb")
    np.save(open_file, y_pred)
    open_file.close

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] == 5:  # Have all 5 grades
        classes = ['1', '2', '3', '4', '5']
    else:
        classes = ['2', '3', '4', '5']
    df_cm = pd.DataFrame(cm, classes, classes)
    fig = plt.figure(figsize=(9, 6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
    balanced_accuracy = round(balanced_accuracy_score(y_true, y_pred), 3)
    class_report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    plt.xlabel("prediction")
    plt.ylabel("label (ground truth)")
    plt.title(f"Confusion Matrix for TestSet:{fold}, Balanced Accuracy:{balanced_accuracy}")
    # logger.info(f"Fold:{fold} Balanced Accuracy:{balanced_accuracy}")
    fname = f"Confusion_Matrix_TestSet{fold}.png"
    plt.savefig(PurePath.joinpath(Path.cwd(), "results", "plots", fname), dpi=fig.dpi)
    plt.close()
    sns.heatmap(pd.DataFrame(class_report).iloc[:-1, :].T, annot=True)
    fname = f"Classification_Report_TestSet{fold}.png"
    plt.savefig(PurePath.joinpath(Path.cwd(), "results", "plots", fname), dpi=fig.dpi)
    plt.close()


def score_avg_classification(cfg, idx, test_df, logger):
    test_ds = ClassificationLoader(cfg, test_df, 'test')
    test_loader = DataLoader(dataset=test_ds, batch_size=cfg.training.dataloader.batch_size, shuffle=True,
                             num_workers=cfg.training.dataloader.num_workers)

    checkpoints = list(Path(PurePath.joinpath(Path.cwd(), "results", "models")).glob("SpineCls_*"))

    total = 0
    correct_preds = 0
    avg_accuracy = []
    models = []
    for checkpoint in checkpoints:
        classification_model, criterion, optimizer, scheduler = build_model(cfg, logger)
        state = torch.load(checkpoint)['model_state']
        classification_model.load_state_dict(state)
        classification_model = DataParallel(classification_model)
        classification_model = classification_model.to('cuda')
        models.append(classification_model)

    with tqdm(models, unit="batch") as tepoch:
        with torch.no_grad():
            tepoch.set_description(f"\t Computing Balanced Accuracy: ")
            for model in tepoch:
                classification_model = model
                classification_model.eval()
                y_true = []
                y_pred = []
                for i_batch, batch in enumerate(test_loader):
                    images = batch['images'].to(cfg.device)
                    labels = batch['labels'].to(cfg.device)

                    outputs = classification_model(images)
                    _, index = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct_preds += (index == labels).sum().item()

                    test_acc = 100 * (correct_preds / total)

                    y_true.append(labels.detach().cpu().numpy().squeeze())
                    y_pred.append(index.detach().cpu().numpy().squeeze())

                    tepoch.set_postfix(test_acc=test_acc)
                avg_accuracy.append(round(balanced_accuracy_score(np.concatenate(y_true), np.concatenate(y_pred)), 3))
            save_cm_kfold(cfg, f"Test Set {idx}", y_true, y_pred, logger)
            avg_model_score = round(np.mean(avg_accuracy), 3)
            logger.info(f"Balanced Test Accuracy for Set {idx} :{round(np.mean(avg_accuracy), 3)}")
    return avg_model_score
