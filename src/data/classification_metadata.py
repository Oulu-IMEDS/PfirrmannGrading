import pandas as pd
from pathlib import PurePath, Path


def build_classification_metadata(cfg, logger):
    df = pd.read_csv(PurePath.joinpath(Path(cfg.mode.classification.dir, cfg.mode.classification.metadata)))
    pfirrmann_col = cfg.mode.classification.pfirrmann_column
    cdf = df.groupby(['patient_id', 'dir_name', pfirrmann_col])['file_name'].count().reset_index(
        name='file_count')

    # Rename the pfirrmann_grade_column for consistency
    cdf.rename(columns={pfirrmann_col: "pfirrmann_grade"}, inplace=True)

    # Added Shuffle 22-04-2022
    cdf = cdf.sample(frac=1).reset_index(drop=True)
    return cdf
