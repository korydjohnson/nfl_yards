

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from network.network import nn1
from network.network import Preprocessor

from scripts.clean_data_rescale import DataCleaner
from scripts.make_features_v2 import FeatureGenerator

from network.convnet import convnet, DataGenerator

device = torch.device('cpu')

print('- read data')
df = pd.read_csv('./input/train.csv', low_memory=False)
folds = pd.read_csv('./input/cvfolds.csv')

cleaner = DataCleaner(df)
df = cleaner.clean_data(df)
df = df.reset_index()

i = 0
playids_tr = folds.loc[(folds['fold'] != i ) & (folds['GameId'].astype(str).str[:4] == '2018'),'PlayId'].values
playids_te = folds.loc[(folds['fold'] == i ) & (folds['GameId'].astype(str).str[:4] == '2018'),'PlayId'].values


df_tr = df[df['PlayId'].isin(playids_tr)].reset_index(drop=True)
df_te = df[df['PlayId'].isin(playids_te)].reset_index(drop=True)

tr = DataGenerator(df_tr, playids_tr)
te = DataGenerator(df_te, playids_te)


## plot

import matplotlib.pyplot as plt
x, y = tr.get_features([0])
x = x[0,:,:,:].detach().numpy()
a = []
for i in range(3): a.append(x[i,:,:])
a = np.stack(a,axis=2)

plt.imshow(a)
plt.show()

## train net





net = convnet(out_channels_1=10, out_channels_2=5, batch_size=32,learning_rate=.0001)

for j in range(100):
    net.train_epochs(nepochs=1, gen=tr)
    loss = net.eval_model(te, batch_size=32)
    print('  - iter %d - loss=%.5f' % (j + 1, loss))






