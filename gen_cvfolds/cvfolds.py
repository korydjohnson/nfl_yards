import numpy as np
import pandas as pd
from random import Random

df = pd.read_csv('./input/trainClean.csv', low_memory=False)

## generate folds
train = df.groupby('PlayId')['GameId'].first().reset_index()
gameids = train['GameId'].unique()
Random(123).shuffle(gameids)
gameids = np.array_split(gameids, 5)

train['fold'] = -1
for i,ids in enumerate(gameids):
    train.loc[train['GameId'].isin(ids),'fold'] = i

train.to_csv('./input/cvfolds.csv', index=False)
