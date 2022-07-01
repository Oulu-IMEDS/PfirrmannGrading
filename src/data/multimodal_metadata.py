import pandas as pd
from pathlib import PurePath, Path


def build_multimodal_metadata(cfg, logger):
    cdf = pd.read_csv(PurePath.joinpath(Path(cfg.mode.multimodal.dir, cfg.mode.multimodal.metadata)))
    logger.info(f'Data Frame Size:{cdf.shape}')

    # target_col = cfg.mode.multimodal.target_column
    # cdf = df.groupby(['patient_id', 'dir_name','sex','vit_d','hba1c','bmi','lbp', 'pfirrmann_grade'])['file_name'].count().reset_index(name='file_count')

    # Added Shuffle 22-04-2022
    cdf = cdf.sample(frac=1).reset_index(drop=True)
    return cdf
