import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import os

def build_segmentation_metadata(cfg,logger):
    images=os.listdir(os.path.join(cfg.data.segmentation.dir,cfg.data.segmentation.images))
    masks=os.listdir(os.path.join(cfg.data.segmentation.dir,cfg.data.segmentation.masks))
    patient_ids=[x.split('_IMG')[0] for x in images]

    #Create and Shuffle the dataframe
    df=pd.DataFrame(data=list(zip(patient_ids,images,masks)),columns=['patient_id','image','mask'])
    segmentation_df=shuffle(df)

    train_split=cfg.data.split_ratio.train_split/100
    val_split=1-cfg.data.split_ratio.val_split/100

    train_df,test_df,val_df=np.split(segmentation_df,[int(train_split*len(segmentation_df)), int(val_split*len(segmentation_df))])

    return train_df,val_df,test_df
