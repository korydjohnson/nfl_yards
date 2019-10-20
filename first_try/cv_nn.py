
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from network.network import nn1

from scripts.clean_data import DataCleaner
from scripts.make_features import FeatureGenerator

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
    print('- Fold %d' %(i+1))

    intr = np.where(df['fold']!=i)[0]
    inte = np.where(df['fold']==i)[0]

    df_tr = df.loc[intr,:].reset_index(drop=True)
    df_te = df.loc[inte,:].reset_index(drop=True)

    print('  - clean data')
    cleaner = DataCleaner(df_tr)
    df_tr = cleaner.clean_data(df_tr)
    df_te = cleaner.clean_data(df_te)

    print('  - get features')
    featgenerator = FeatureGenerator(df_tr)
    xtr, ytr, playid_tr = featgenerator.make_features(df_tr)
    xte, yte, playid_te = featgenerator.make_features(df_te)

    print('  - initialize network')
    feat_num = ['YardLine', 'Distance', 'DefendersInTheBox', 'Week']
    feat_cat = [
        'Quarter', 'PossessionTeam', 'Down',
        'OffenseFormation', 'OffensePersonnel', 'DefensePersonnel',
        'HomeTeamAbbr', 'VisitorTeamAbbr', 'StadiumType', 'Turf', 'GameWeather',
    ]

    scaler = StandardScaler()
    scaler.fit(xtr[feat_num])
    xtr[feat_num] = scaler.transform(xtr[feat_num])
    xte[feat_num] = scaler.transform(xte[feat_num])

    xtr_num = torch.from_numpy(xtr[feat_num].values).float().to(device)
    xte_num = torch.from_numpy(xte[feat_num].values).float().to(device)

    n_unique = xtr[feat_cat].max().values + 1
    xtr_cat = torch.from_numpy(xtr[feat_cat].values).long().to(device)
    xte_cat = torch.from_numpy(xte[feat_cat].values).long().to(device)

    ytr = pd.get_dummies(pd.Categorical(ytr.iloc[:,0] + 99, range(199))).cumsum(axis=1).values.astype(float)
    yte = pd.get_dummies(pd.Categorical(yte.iloc[:,0] + 99, range(199))).cumsum(axis=1).values.astype(float)
    ytr = torch.from_numpy(ytr).float().to(device)
    yte = torch.from_numpy(yte).float().to(device)

    xtr = [xtr_num, xtr_cat]
    xte = [xte_num, xte_cat]

    input_size = len(feat_num) + 2 * len(feat_cat)
    net = nn1(input_size=input_size, hidden_size=25, output_size=199, batch_size=256, n_unique=n_unique).to(device)

    # for j in range(100):
    #     net.train_epochs(nepochs=1, X=xtr, y=ytr)
    #     loss = net.eval_model(xte, yte)
    #     print('  - iter %d - loss=%.5f' % (j+1, loss))

    print('  - train network')
    net.train_epochs(nepochs=50, X=xtr, y=ytr)
    loss = net.eval_model(xte, yte)
    print('  - CRPS=%.5f' % (loss))

    observed.append(yte.cpu().detach().numpy())
    predicted.append(net.predict(xte).cpu().detach().numpy())
    playids.append(playid_te)


print('- compute overall CV score')
observed = np.concatenate(observed)
predicted = np.concatenate(predicted)
playids = np.concatenate(playids)

loss = np.mean((observed-predicted)**2)
print('- CRPS=%.5f' %(loss))

print('Done!!!')

