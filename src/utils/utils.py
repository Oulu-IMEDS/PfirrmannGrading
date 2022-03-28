import matplotlib.pyplot as plt
from omegaconf import DictConfig
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import hydra
import cv2
import torch

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

def visualize_random_img_masks(image_batch,logger):
    index = np.random.randint(0, len(image_batch))
    #logger.info(image_batch.__getitem__(0))
    image_1 = image_batch.__getitem__(index)['transformed_raw'][0]
    mask_1 = image_batch.__getitem__(index)['transformed_mask'][0]
    plt.subplot(1, 2, 1)
    plt.imshow(image_1,cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(mask_1,cmap='gray')
    plt.show()
    plt.close()

def img_to_tensor(img_raw):
    tensor_img=A.Compose([A.Resize(512, 512), ToTensorV2()])
    transformed_img=tensor_img(image=img_raw)['image']
    return transformed_img

def visualize_segmentation(cfg):
    test_file='/data/nfbc_segmentation/IMG010166_0013_00005.png'
    model_checkpoint='/home/nash/PycharmProjects/PfirrmannGrading/results/models/SpineSeg_Epoch14_EpochIOU0.9235.pth'
    state = torch.load(model_checkpoint)['model_state']  # Change to model
    img_raw = cv2.imread(test_file)
    img_torch=img_to_tensor(img_raw)
    print(img_torch.shape)
    img_torch = img_torch.unsqueeze(0).float().to('cuda')

    print(img_torch.shape)

    model = hydra.utils.instantiate(cfg.model.Unet).to('cuda')
    model.load_state_dict(state)
    model = model.to('cuda')
    model.eval()


    pred = 0
    pred += model(img_torch)
    pred = pred.squeeze().argmax(0).to('cpu').numpy()


    plt.figure(figsize=(30, 10))
    plt.subplot(141)
    plt.imshow(img_raw)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(142)
    plt.imshow(pred)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(143)
    img_contours = img_raw.copy()
    for cls_id in range(1, 14):
        cur_mask = np.uint8(pred == cls_id) * 255
        if cur_mask.sum() == 0:
            continue
        contours, hierarchy = cv2.findContours(cur_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if cls_id % 2:
            cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)
    plt.imshow(img_contours)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(144)
    img_contours = img_raw.copy()
    for cls_id in range(1, 14):
        cur_mask = np.uint8(pred == cls_id) * 255
        if cur_mask.sum() == 0:
            continue
        contours, hierarchy = cv2.findContours(cur_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if cls_id % 2 == 0:
            cv2.drawContours(img_contours, contours, -1, (255, 0, 0), 3)

    plt.imshow(img_contours)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.close()