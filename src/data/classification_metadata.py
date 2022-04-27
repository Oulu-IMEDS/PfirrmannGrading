import pandas as pd
from pathlib import PurePath, Path


def build_classification_metadata(cfg, logger):
    df = pd.read_csv(PurePath.joinpath(Path(cfg.mode.classification.dir, cfg.mode.classification.metadata)))
    cdf = df.groupby(['patient_id', 'dir_name', 'pfirrmann_grade'])['file_name'].count().reset_index(name='file_count')
    # Added Shuffle 22-04-2022
    cdf = cdf.sample(frac=1).reset_index(drop=True)
    return cdf
