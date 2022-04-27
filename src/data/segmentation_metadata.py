import pandas as pd
from pathlib import PurePath, Path
from sklearn.utils import shuffle


def build_segmentation_metadata(cfg, logger):
    images = [fname.name for fname in
              list(PurePath.joinpath(Path(cfg.mode.input.dir, cfg.mode.input.images)).glob("*.png"))]
    masks = [fname.name for fname in
             list(PurePath.joinpath(Path(cfg.mode.input.dir, cfg.mode.input.masks)).glob("*.png"))]
    patient_ids = [x.split('_IMG')[0] for x in images]

    # Create and Shuffle the dataframe
    df = pd.DataFrame(data=list(zip(patient_ids, images, masks)), columns=['patient_id', 'image', 'mask'])
    segmentation_df = shuffle(df)

    return segmentation_df
