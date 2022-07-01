import albumentations as A
import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import torch
from albumentations.pytorch import ToTensorV2
from pathlib import Path, PurePath
from sklearn.model_selection import GroupShuffleSplit
from src.model.model_architectures import build_model
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from src.data.segmentation_loader import SegmentationLoader


def score_avg_segmentation(cfg, test_df, logger):
    test_ds = SegmentationLoader(cfg, test_df, 'test')
    test_loader = DataLoader(dataset=test_ds, batch_size=cfg.training.dataloader.batch_size,
                             num_workers=cfg.training.dataloader.num_workers)

    checkpoints = list(Path(PurePath.joinpath(Path.cwd(), "results", "models")).glob("SpineSeg_*"))

    models = []
    iou_cum = []
    for checkpoint in checkpoints:
        segmentation_model, criterion, optimizer, scheduler = build_model(cfg, logger)
        state = torch.load(checkpoint)['model_state']
        segmentation_model.load_state_dict(state)
        segmentation_model = DataParallel(segmentation_model)
        segmentation_model = segmentation_model.to('cuda')
        models.append(segmentation_model)

    with tqdm(models, unit="batch") as tepoch:
        with torch.no_grad():
            tepoch.set_description(f"\t Computing IOU: ")
            for model in tepoch:
                segmentation_model = model
                segmentation_model.eval()

                for i_batch, batch in enumerate(test_loader):
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
                    tepoch.set_postfix(mean_iou=np.mean(iou_cum))

    return np.mean(iou_cum)


def split_test_train(cfg, df):
    # Split the main df into train,test splits
    split_mode = GroupShuffleSplit(test_size=cfg.training.train_test.test_perc, n_splits=1,
                                   random_state=cfg.random_seed)
    data_split = split_mode.split(df, groups=df['patient_id'])
    train_indices, test_indices = next(data_split)

    segmentation_df = df.iloc[train_indices]
    segmentation_df = segmentation_df.reset_index()
    test_df = df.iloc[test_indices]
    test_df = test_df.reset_index()
    return segmentation_df, test_df


def calculate_confusion_matrix_from_arrays(prediction, ground_truth, nr_labels):
    """
    https://github.com/ternaus/robot-surgery-segmentation/blob/master/validation.py
    """

    replace_indices = np.vstack((
        ground_truth.flatten(),
        prediction.flatten())
    ).T

    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(nr_labels, nr_labels),
        range=[(0, nr_labels), (0, nr_labels)]
    )
    confusion_matrix = confusion_matrix.astype(np.uint64)
    return confusion_matrix


def calculate_iou(confusion_matrix):
    """
    https://github.com/ternaus/robot-surgery-segmentation/blob/master/validation.py
    """
    confusion_matrix = confusion_matrix.astype(float)
    ious = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = true_positives + false_positives + false_negatives
        if denom == 0:
            iou = 0
        else:
            iou = float(true_positives) / denom
        ious.append(iou)
    return ious


def visualize_random_img_masks(image_batch, writer, logger):
    plt.figure(figsize=(20, 20))
    idx, imgs = next(enumerate(image_batch))
    plot_ind = 1
    for index in range(8):
        image_1 = imgs['transformed_raw'][index].unsqueeze(0)
        mask = imgs['transformed_mask'][index].unsqueeze(0)

        # Inverse Normalize the image before displaying
        inv_normalize = transforms.Normalize(
            mean=[-m / s for m, s in zip([0.181], [0.184])],
            std=[1 / s for s in [0.184]]
        )

        inv_tensor = inv_normalize(image_1)

        plt.subplot(4, 4, plot_ind)
        plt.imshow(inv_tensor.squeeze(0).permute(1, 2, 0))
        plot_ind += 1
        plt.subplot(4, 4, plot_ind)
        plt.imshow(mask.permute(1, 2, 0))
        plot_ind += 1
    plt.title(f"Random Images from Train Set")
    plt.savefig('train_images.png', dpi='figure')
    img_raw = cv2.imread(str('train_images.png'))
    images_train = torch.tensor(cv2.cvtColor(img_raw, cv2.IMREAD_COLOR))
    writer.add_image(f"Random Images from Train Set", images_train, dataformats='HWC')
    plt.close()


def img_to_tensor(cfg, img_raw, tensor=False):
    if tensor:
        tensor_img = A.Compose([A.Resize(cfg.mode.crop.width, cfg.mode.crop.height), ToTensorV2()])
    else:
        tensor_img = A.Compose([A.Resize(cfg.mode.crop.width, cfg.mode.crop.height)])
    transformed_img = tensor_img(image=img_raw)['image']
    return transformed_img


def get_spine_unit_rect(cfg, input_image, model_prediction, spine_unit, logger, pad=None, display=False):
    if spine_unit == 'l1-l2':
        s_unit = (np.uint8(model_prediction == cfg.mode.spine_segments.l1) + np.uint8(
            model_prediction == cfg.mode.spine_segments.d1) + np.uint8(
            model_prediction == cfg.mode.spine_segments.l2)) * 255
    elif spine_unit == 'l2-l3':
        s_unit = (np.uint8(model_prediction == cfg.mode.spine_segments.l2) + np.uint8(
            model_prediction == cfg.mode.spine_segments.d2) + np.uint8(
            model_prediction == cfg.mode.spine_segments.l3)) * 255
    elif spine_unit == 'l3-l4':
        s_unit = (np.uint8(model_prediction == cfg.mode.spine_segments.l3) + np.uint8(
            model_prediction == cfg.mode.spine_segments.d3) + np.uint8(
            model_prediction == cfg.mode.spine_segments.l4)) * 255
    elif spine_unit == 'l4-l5':
        s_unit = (np.uint8(model_prediction == cfg.mode.spine_segments.l4) + np.uint8(
            model_prediction == cfg.mode.spine_segments.d4) + np.uint8(
            model_prediction == cfg.mode.spine_segments.l5)) * 255
    elif spine_unit == 'l5-s1':
        s_unit = (np.uint8(model_prediction == cfg.mode.spine_segments.s1) + np.uint8(
            model_prediction == cfg.mode.spine_segments.d5) + np.uint8(
            model_prediction == cfg.mode.spine_segments.l5)) * 255
    elif spine_unit == 'd1':
        s_unit = np.uint8(model_prediction == cfg.mode.spine_segments.d1) * 255
    elif spine_unit == 'd2':
        s_unit = np.uint8(model_prediction == cfg.mode.spine_segments.d2) * 255
    elif spine_unit == 'd3':
        s_unit = np.uint8(model_prediction == cfg.mode.spine_segments.d3) * 255
    elif spine_unit == 'd4':
        s_unit = np.uint8(model_prediction == cfg.mode.spine_segments.d4) * 255
    elif spine_unit == 'd5':
        s_unit = np.uint8(model_prediction == cfg.mode.spine_segments.d5) * 255
    elif spine_unit == 'l1':
        s_unit = np.uint8(model_prediction == cfg.mode.spine_segments.l1) * 255
    elif spine_unit == 'l2':
        s_unit = np.uint8(model_prediction == cfg.mode.spine_segments.l2) * 255
    elif spine_unit == 'l3':
        s_unit = np.uint8(model_prediction == cfg.mode.spine_segments.l3) * 255
    elif spine_unit == 'l4':
        s_unit = np.uint8(model_prediction == cfg.mode.spine_segments.l4) * 255
    elif spine_unit == 'l5':
        s_unit = np.uint8(model_prediction == cfg.mode.spine_segments.l5) * 255
    elif spine_unit == 's1':
        s_unit = np.uint8(model_prediction == cfg.mode.spine_segments.s1) * 255

    contours, hierarchy = cv2.findContours(s_unit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the area of contours to determine if the slice is worth looking into for
    if contours:
        big_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(big_contour)
        # logger.info(f"Contour Area:{contour_area} Unit:{spine_unit}")
        # If the code is not able to detect contours it could be a bad slice to use
        if (spine_unit in ('l1-l2', 'l2-l3', 'l3-l4', 'l4-l5', 'l5-s1') and contour_area < 3000):
            return 0
        else:
            x, y, width, height = cv2.boundingRect(big_contour)
            result = input_image[y:y + height, x:x + width]

            dw = cfg.mode.spine_segments.crop_to_width - result.shape[0]
            dh = cfg.mode.spine_segments.crop_to_height - result.shape[1]

            if pad:
                # Apply a white border around the segmented area
                top = max(0, dw // 2)
                bottom = max(0, dw - top)
                left = max(0, dh // 2)
                right = max(0, dh - left)
                result = cv2.copyMakeBorder(result, top, bottom, left, right,
                                            cv2.BORDER_CONSTANT, value=(255, 255, 255))

            # If resultant crop has dimensions more than intended crop dimensions, apply resize
            if (result.shape[0], result.shape[1]) != (
                    cfg.mode.spine_segments.crop_to_width, cfg.mode.spine_segments.crop_to_height):
                result = cv2.resize(result,
                                    (cfg.mode.spine_segments.crop_to_width, cfg.mode.spine_segments.crop_to_height),
                                    interpolation=cv2.INTER_AREA)
        if display:
            plt.imshow(result)
            plt.show()
            plt.close()
        else:
            return result


def model_prediction(cfg, input):
    parent = "/home/nash/PycharmProjects/PfirrmannGrading/results/models"
    # checkpoints = list(Path(PurePath.joinpath(Path.cwd(), "results", "models")).glob("SpineSeg*"))
    checkpoints = list(Path(parent).glob("SpineSeg*"))
    img_raw = cv2.imread(input)
    img_torch = img_to_tensor(cfg, img_raw, tensor=True)
    img_raw = img_to_tensor(cfg, img_raw)
    img_torch = img_torch.unsqueeze(0).float().to('cuda')

    prediction = 0
    for checkpoint in checkpoints:
        model_checkpoint = checkpoint
        model = hydra.utils.instantiate(cfg.segmentation_architecture).to('cuda')
        state = torch.load(model_checkpoint)['model_state']
        model.load_state_dict(state)
        model = DataParallel(model)
        model = model.to('cuda')
        model.eval()
        prediction += model(img_torch)

    prediction /= len(checkpoints)
    prediction = prediction.squeeze().argmax(0).to('cpu').numpy()

    return img_raw, prediction


def visualize_segmentation(cfg, input, logger):
    original_image, prediction = model_prediction(cfg, input)

    plt.figure(figsize=(30, 10))
    plt.subplot(141)
    plt.imshow(original_image)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(142)
    plt.imshow(prediction)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(143)
    img_contours = original_image.copy()
    for cls_id in range(1, cfg.mode.input.n_classes):
        cur_mask = np.uint8(prediction == cls_id) * 255
        if cur_mask.sum() == 0:
            continue
        contours, hierarchy = cv2.findContours(cur_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if cls_id % 2:
            cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)
    plt.imshow(img_contours)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(144)
    img_contours = original_image.copy()
    for cls_id in range(1, 14):
        cur_mask = np.uint8(prediction == cls_id) * 255
        if cur_mask.sum() == 0:
            continue
        contours, hierarchy = cv2.findContours(cur_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if cls_id % 2 == 0:
            cv2.drawContours(img_contours, contours, -1, (255, 0, 0), 3)

    plt.imshow(img_contours)
    plt.xticks([])
    plt.yticks([])
    plt.show()

    get_spine_unit_rect(cfg, original_image, prediction, 'l3-l4', logger, pad=None, display=True)

    plt.close()


# Overlays segmentations on the actual image and generates an image with original image and overlayed map
def generate_spine_map(cfg, logger):
    for img_file in tqdm(list(PurePath.joinpath(Path(cfg.mode.test.dir)).glob("*.png")), ncols=50,
                         desc="Generating Predictions"):
        input_raw, prediction = model_prediction(cfg, str(img_file))
        plt.figure(figsize=(10, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(input_raw)
        vb_units = np.uint8(prediction == cfg.mode.spine_segments.s1) + np.uint8(
            prediction == cfg.mode.spine_segments.l5) + np.uint8(prediction == cfg.mode.spine_segments.l4) + np.uint8(
            prediction == cfg.mode.spine_segments.l3) + np.uint8(prediction == cfg.mode.spine_segments.l2) + np.uint8(
            prediction == cfg.mode.spine_segments.l1) + np.uint8(prediction == cfg.mode.spine_segments.t12) * 255
        disc_units = np.uint8(prediction == cfg.mode.spine_segments.d5) + np.uint8(
            prediction == cfg.mode.spine_segments.d4) + np.uint8(prediction == cfg.mode.spine_segments.d3) + np.uint8(
            prediction == cfg.mode.spine_segments.d2) + np.uint8(prediction == cfg.mode.spine_segments.d1) + np.uint8(
            prediction == cfg.mode.spine_segments.d12) * 255
        contours, hierarchy = cv2.findContours(vb_units, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(input_raw, contours, -1, (0, 255, 0), 3)
        contours, hierarchy = cv2.findContours(disc_units, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(input_raw, contours, -1, (0, 0, 255), 3)
        plt.subplot(1, 2, 2)
        plt.imshow(input_raw)
        plt.savefig(cfg.mode.test.dir + "/" + img_file.name + "_pred.png")
        plt.close()


def categorize_spine_units(cfg, mri_labels, mri_file, logger, spine_units):
    disc_mapping = {'l1-l2': 'l1_l2', 'l2-l3': 'l2_l3',
                    'l3-l4': 'l3_l4', 'l4-l5': 'l4_l5',
                    'l5-s1': 'l5_s1'}
    # Get model prediction for the inputs file
    image_raw, prediction = model_prediction(cfg, str(mri_file))
    patient_meta_data = []
    for unit in spine_units:
        # result = get_spine_unit(cfg, image_raw, prediction, unit, logger)
        result = get_spine_unit_rect(cfg, image_raw, prediction, unit, logger)

        # logger.info(result.shape)
        # if we are unable to find spine unit for
        if np.any(result):
            pfirrmann_grade = mri_labels[disc_mapping[unit]].values[0]

            dir_name = mri_labels['patient_id'].values[0].replace("+", "_") + "_" + unit + "_" + "PG" + str(
                pfirrmann_grade)
            file_name = mri_file.name.replace(".png", "_" + unit + "_PG" + str(pfirrmann_grade) + "_" + ".png")
            # logger.info(f"Spine Unit:{unit} Pfirrmann Grade:{pfirrmann_grade} Dir Name:{dir_name} File Name:{file_name}")
            disc_path = Path(cfg.mode.mri_scans.target_root, "NFBC_Spine_Unit_Classification", dir_name)
            disc_path.mkdir(parents=True, exist_ok=True)
            # if mri_labels['patient_id'].values[0]=='010166+0013' and unit=='l1-l2':
            #    plt.imshow(result)
            #    plt.show()

            target_file = PurePath(cfg.mode.mri_scans.target_root, "NFBC_Spine_Unit_Classification",
                                   dir_name, file_name)
            # Categorize the image into corresponding pfirrmann grade folder
            cv2.imwrite(str(target_file), result)
            patient_meta_data.append(
                [mri_labels['patient_id'].values[0], dir_name, file_name, unit, pfirrmann_grade,
                 mri_labels['sex'].values[0],
                 mri_labels['vit_d'].values[0], mri_labels['hba1c'].values[0], mri_labels['bmi'].values[0],
                 mri_labels['lbp'].values[0]])
    return patient_meta_data


def generate_mri_labels(cfg, logger):
    spine_units = ['l1-l2', 'l2-l3', 'l3-l4', 'l4-l5', 'l5-s1']
    target_root = cfg.mode.mri_scans.target_root
    # Make directories with pfirrmann grade names to move spine units
    parent_path = Path(target_root, "NFBC_Spine_Unit_Classification")
    parent_path.mkdir(parents=True, exist_ok=True)

    # Read the pfirrmann grade labels for patients
    mri_labels = pd.read_csv(cfg.mode.mri_scans.labels)
    patient_ids = mri_labels.patient_id.values

    # patient_id_subset = random.choices(patient_ids, k=1)
    # pdf = pd.DataFrame(data=patient_id_subset, columns=['patient_id'])
    # pdf.to_csv(str(parent_path) + "/patient_id_subset.csv")

    # Path where the MRI images for the corresponding labels are present
    mri_path = cfg.mode.mri_scans.dir
    total_count = 0
    meta_data = pd.DataFrame()
    with tqdm(patient_ids, unit="batch") as patient_ids:
        for patient_id in patient_ids:
            patient_ids.set_description(f"Processing Patient ID: {patient_id}")
            wildcard = "IMG" + patient_id.replace("+", "_") + "*"
            mri_count = 0
            for mri in list(PurePath.joinpath(Path(mri_path)).glob(wildcard)):
                # logger.info(mri)
                mri_count += 1
                total_count += 1
                patient_data = categorize_spine_units(cfg, mri_labels[mri_labels['patient_id'] == patient_id], mri,
                                                      logger, spine_units)
                patient_meta = pd.DataFrame(data=patient_data,
                                            columns=['patient_id', 'dir_name', 'file_name', 'spine_unit',
                                                     'pfirrmann_grade',
                                                     'sex', 'vit_d', 'hba1c', 'bmi', 'lbp'])
                meta_data = pd.concat([patient_meta, meta_data])
                patient_ids.set_postfix(mri_count=mri_count, total_count=total_count)
        meta_data.to_csv(str(parent_path) + "/patient_meta_data.csv", index=False)
