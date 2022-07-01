import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from pathlib import Path, PurePath
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, matthews_corrcoef
from sklearn.model_selection import GroupShuffleSplit
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from tqdm import tqdm
from src.training.focal_loss import FocalLoss
from src.data.classification_loader import ClassificationLoader
from src.model.model_architectures import build_model


def split_train_test(cfg, df):
    # Split the main df into train,test splits
    split_mode = GroupShuffleSplit(test_size=cfg.training.train_test.test_perc, n_splits=1,
                                   random_state=cfg.random_seed)
    data_split = split_mode.split(df, groups=df['patient_id'])
    train_indices, test_indices = next(data_split)

    classification_df = df.iloc[train_indices]
    classification_df = classification_df.reset_index()
    test_df = df.iloc[test_indices]
    test_df = test_df.reset_index()
    return classification_df, test_df


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


def compute_lins_ccc(cfg, y_true, y_pred):
    pearson_cor = np.corrcoef(y_true, y_pred)[0][1]
    # Compute mean of true and predicted
    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)
    # Compute variance of true and predicted
    y_true_var = np.var(y_true)
    y_pred_var = np.var(y_pred)
    # Compute standard deviation of true and pred
    y_true_stdev = np.std(y_true)
    y_pred_stdev = np.std(y_pred)
    # Compute Lin's Concordance Correlation Coefficient
    numerator = 2 * pearson_cor * y_true_stdev * y_pred_stdev
    denominator = y_true_var + y_pred_var + (y_true_mean - y_pred_mean) ** 2
    result = numerator / denominator
    return result


def save_cm_kfold(cfg, fold, y_true, y_pred, tta, logger):
    # Save confusion matrix results per fold
    y_true = np.concatenate(y_true)

    if not tta:
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
    classes = ['2', '3', '4', '5']
    df_cm = pd.DataFrame(cm, classes, classes)
    fig = plt.figure(figsize=(9, 6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='Blues')
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    cohen_kappa = cohen_kappa_score(y_true, y_pred)
    matthew_cc = matthews_corrcoef(y_true, y_pred)
    lins_ccc = compute_lins_ccc(cfg, y_true, y_pred)
    class_report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    plt.xlabel("prediction")
    plt.ylabel("label (ground truth)")
    plt.title(f"Confusion Matrix for:{fold}, Accuracy:{accuracy:.3f} Balanced Accuracy:{balanced_accuracy:.3f}")
    logger.info(f"Seed:{cfg.random_seed},Accuracy:{accuracy:.3f},Balanced Accuracy:{balanced_accuracy:.3f},Cohen's "
                f"Kappa:{cohen_kappa:.3f},Matthew_CC:{matthew_cc:.3f},Lin's CCC:{lins_ccc:.3f}")
    fname = f"Confusion_Matrix_TestSet{fold}.png"
    plt.savefig(PurePath.joinpath(Path.cwd(), "results", "plots", fname), dpi=fig.dpi)
    plt.close()
    sns.heatmap(pd.DataFrame(class_report).iloc[:-1, :].T, annot=True)
    fname = f"Classification_Report_TestSet{fold}.png"
    plt.savefig(PurePath.joinpath(Path.cwd(), "results", "plots", fname), dpi=fig.dpi)
    plt.close()


def save_mis_classifications(cfg, dir_names, y_true, y_pred):
    # pred = []
    mc = []
    class_idx = {0: '2', 1: '3', 2: '4', 3: '5'}
    odf = pd.read_csv(PurePath.joinpath(Path(cfg.mode.classification.dir, cfg.mode.classification.metadata)))
    for idx, (e1, e2) in enumerate(zip(y_true, y_pred)):
        odf.loc[odf['dir_name'] == dir_names[idx], 'pred'] = class_idx[e2]
        # pred.append([dir_names[idx], class_idx[e1], class_idx[e2]])
        if e1 != e2:
            mc.append([dir_names[idx], class_idx[e1], class_idx[e2]])
    df = pd.DataFrame(data=mc, columns=['dir_name', 'y_true', 'y_pred'])
    # pred_df = pd.DataFrame(data=pred, columns=['dir_name', 'y_true', 'y_pred'])
    # Save the misclassifications
    odf['pred'].fillna(0, inplace=True)
    odf['pred'] = odf['pred'].astype('int')
    odf = odf[odf['pred'] != 0]
    odf.to_csv('classifications.csv', index=False)
    # pred_df.to_csv('classfications.csv', index=False)
    plt.figure(figsize=(30, 30))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        image_list = sorted(list(Path(cfg.mode.classification.dir, df.iloc[i]['dir_name']).glob("*.png")))
        sagittal = cv2.imread(str(image_list[len(image_list) // 2]))
        plt.imshow(sagittal)
        plt.title(f"File:{df.iloc[i]['dir_name']} Consensus:{df.iloc[i]['y_true']}, Predicted:{df.iloc[i]['y_pred']}")
        plt.xticks([])
        plt.yticks([])
    plt.savefig('misclassifications.png', dpi='figure')
    plt.close()


def score_avg_classification(cfg, test_df, logger):
    logger.info(f"Test Set:{test_df.shape}")
    test_ds = ClassificationLoader(cfg, test_df, 'test')
    test_loader = DataLoader(dataset=test_ds, batch_size=cfg.training.dataloader.batch_size,
                             num_workers=cfg.training.dataloader.num_workers)

    checkpoints = list(Path(PurePath.joinpath(Path.cwd(), "results", "models")).glob("SpineCls_*"))
    # checkpoints = list(Path("/home/nash/PycharmProjects/PfirrmannGrading/outputs/2022-05-31/11-35-54/results/models
    # ").glob("SpineCls_*"))

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
                dirs = []
                for i_batch, batch in enumerate(test_loader):
                    images = batch['images'].to(cfg.device)
                    labels = batch['labels'].to(cfg.device)
                    dir_names = batch['dir_name']

                    outputs = classification_model(images)
                    _, index = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct_preds += (index == labels).sum().item()

                    test_acc = 100 * (correct_preds / total)

                    dirs.append(dir_names)
                    if labels.detach().cpu().numpy().squeeze().size > 1:
                        y_true.append(labels.detach().cpu().numpy().squeeze())
                        y_pred.append(index.detach().cpu().numpy().squeeze())
                    else:
                        y_true.append([labels.detach().cpu().numpy().squeeze()])
                        y_pred.append([index.detach().cpu().numpy().squeeze()])

                    tepoch.set_postfix(test_acc=test_acc)
                avg_accuracy.append(balanced_accuracy_score(np.concatenate(y_true), np.concatenate(y_pred)))
            save_cm_kfold(cfg, f"Test Set", y_true, y_pred, logger)
            save_mis_classifications(cfg, np.concatenate(dirs), np.concatenate(y_true), np.concatenate(y_pred))
            avg_model_score = np.mean(avg_accuracy)
    return avg_model_score


def score_avg_classification_score_tta(cfg, test_df, logger):
    logger.info(f"Test Set:{test_df.shape}")
    test_ds = ClassificationLoader(cfg, test_df, 'train')
    test_loader = DataLoader(dataset=test_ds, batch_size=cfg.training.testing.batch_size,
                             num_workers=cfg.training.dataloader.num_workers)

    checkpoints = list(Path(PurePath.joinpath(Path.cwd(), "results", "models")).glob("SpineCls_*"))
    # checkpoints = list(Path("/home/nash/PycharmProjects/PfirrmannGrading/multirun/2022-06-28/15-48-42/0/results
    # /models").glob("SpineCls_*"))

    total = 0
    correct_preds = 0
    avg_accuracy = []
    models = []
    for checkpoint in checkpoints:
        multimodal_classifier, criterion, optimizer, scheduler = build_model(cfg, logger)
        state = torch.load(checkpoint)['model_state']
        multimodal_classifier.load_state_dict(state)
        multimodal_classifier = DataParallel(multimodal_classifier)
        multimodal_classifier = multimodal_classifier.to('cuda')
        models.append(multimodal_classifier)

    for model in models:
        multimodal_classifier = model
        multimodal_classifier.eval()
        tta = []
        d = []
        with tqdm(range(cfg.training.tta), unit="tta_run") as tta_run:
            with torch.no_grad():
                tta_run.set_description(f"\t Computing Balanced Accuracy using TTA: ")
                for _ in tta_run:
                    y_true = []
                    dirs = []
                    preds = None
                    with torch.no_grad():
                        for i_batch, batch in enumerate(test_loader):
                            images = batch['images'].to(cfg.device)
                            labels = batch['labels'].to(cfg.device)
                            dir_names = batch['dir_name']

                            outputs = multimodal_classifier(images)
                            if i_batch == 0:
                                preds = outputs.detach().cpu().numpy()
                            else:
                                preds = np.vstack((preds, outputs.detach().cpu().numpy()))
                            _, index = torch.max(outputs, 1)
                            total += labels.size(0)
                            correct_preds += (index == labels).sum().item()

                            test_acc = 100 * (correct_preds / total)

                            dirs.append(dir_names)
                            if labels.detach().cpu().numpy().squeeze().size > 1:
                                y_true.append(labels.detach().cpu().numpy().squeeze())
                            else:
                                y_true.append([labels.detach().cpu().numpy().squeeze()])

                            tta_run.set_postfix(test_acc=test_acc)
                    tta.append(preds)
                    d.append(dirs)
        y_pred = np.argmax(np.sum(tta, axis=0) / cfg.training.tta, axis=1)
        avg_accuracy.append(balanced_accuracy_score(np.concatenate(y_true), y_pred))
    save_cm_kfold(cfg, f"Test Set", y_true, y_pred, True, logger)
    save_mis_classifications(cfg, np.concatenate(d[0]), np.concatenate(y_true), y_pred)
    avg_model_score = np.mean(avg_accuracy)
    return avg_model_score
