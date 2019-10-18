
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from network.network import nn1

from eda_Py.clean_data import DataCleaner
from eda_Py.make_features import FeatureGenerator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('- read data')
df = pd.read_csv('./input/train.csv', low_memory=False)
folds = pd.read_csv('./input/cvfolds.csv')

print('- merge cv folds to train data')
df = df.merge(folds[['PlayId', 'fold']], on = 'PlayId', how = 'left')


observed = []
predicted = []
playids = []
nfolds = 5
for i in range(nfolds):
    print('- Fold %d' %(i))

    intr = np.where(df['fold']!=i)[0]
    inte = np.where(df['fold']==i)[0]

    df_tr = df.loc[intr,:].reset_index(drop=True)
    df_te = df.loc[inte,:].reset_index(drop=True)

    print('  - clean data')
    cleaner = DataCleaner(df_tr)
    df_tr = cleaner.clean_data(df_tr)
    df_te = cleaner.clean_data(df_te)

    print('  - get features')
    #featgenerator = FeatureGenerator(df_tr)
    #xtr, ytr, playid_tr = featgenerator.make_features(df_tr)
    #xte, yte playid_te = featgenerator.make_features(df_te)

    print('  - initialize network')
    #input_size = len(feat_num) + 2 * len(feat_cat)
    #net = nn1(input_size=input_size, hidden_size=25, output_size=199, batch_size=256, n_unique=n_unique).to(device)

    print('  - train network')
    #net.train_epochs(nepochs=100, X=xtr, y=ytr)
    #loss = net.eval_model(xte, yte)
    #print('  - iter %d - loss=%.f' % (i, loss))

    #observed.append(yte)
    #predicted.append(net.predict(xte))
    #playids.append(playid_te)

score = np.mean((y-pred)**2)
print('-  CV-Score=%.4f' %(score))

