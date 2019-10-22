
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from network.network import nn1, nn2
from network.network import Preprocessor

from scripts.clean_data import DataCleaner
from scripts.make_features import FeatureGenerator

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

print('- read data')
df = pd.read_csv('./input/train.csv', low_memory=False)
folds = pd.read_csv('./input/cvfolds.csv')

print('- merge cv folds to train data')
df = df.merge(folds[['PlayId', 'fold']], on = 'PlayId', how = 'left')

feat_num = [
    'Distance', 'DefendersInTheBox', 'Week', 'DistanceToGoal',
    'RusherInfo_Acc', 'RusherInfo_DistDef', 'RusherInfo_DistGoal',
    'RusherInfo_DistLOS',
    'RusherInfo_SpeedX', 'RusherInfo_SpeedY', 'RusherInfo_DistDefvsOff', 'RusherInfo_DistOffMean',
    'RusherInfo_DistDefMean', 'RusherInfo_AccClosestvsRusher', 'RusherInfo_SpeedClosestvsRusher',
    'RusherInfo_DefXStd', 'RusherInfo_DefYStd', 'RusherInfo_OffXStd', 'RusherInfo_OffYStd',
    'RusherInfo_OffCountWR',
]
feat_cat = [
    'Quarter', 'PossessionTeam', 'Down',
    'OffenseFormation', 'OffensePersonnel', 'DefensePersonnel',
    'HomeTeamAbbr', 'VisitorTeamAbbr', 'StadiumType', 'Turf', 'GameWeather',
    'RusherInfo_Pos',
]

observed = []
predicted = []
playids = []
nfolds = 5
for i in range(nfolds):
    print('- Fold %d' %(i+1))

    intr = np.where((df['fold']!=i) & (df['GameId'].astype(str).str[:4] == '2018'))[0]
    inte = np.where((df['fold']==i) & (df['GameId'].astype(str).str[:4] == '2018'))[0]

    df_tr = df.loc[intr,:].reset_index(drop=True)
    df_te = df.loc[inte,:].reset_index(drop=True)

    print('  - clean data')
    cleaner = DataCleaner(df_tr)
    df_tr = cleaner.clean_data(df_tr)
    df_te = cleaner.clean_data(df_te)

    print('  - get features')
    featgenerator = FeatureGenerator()
    xtr, ytr, playid_tr = featgenerator.make_features(df_tr, test=False)
    xte, yte, playid_te = featgenerator.make_features(df_te, test=False)

    print('  - transform features')
    preproc = Preprocessor(xtr, feat_num, feat_cat)
    xtr, ytr = preproc.transform_data(xtr, ytr,  test=False)
    xte, yte = preproc.transform_data(xte, yte, test=False)
    n_unique = xtr[1].max(0).values.detach().numpy() + 1

    # input_size = len(feat_num) + 2 * len(feat_cat)
    # net = nn1(input_size=input_size, hidden_size=25, output_size=199, batch_size=128, n_unique=n_unique).to(device)
    #
    # for j in range(100):
    #     net.train_epochs(nepochs=1, X=xtr, y=ytr)
    #     loss = net.eval_model(xte, yte)
    #     print('  - iter %d - loss=%.5f' % (j+1, loss))

    print('  - train network')
    nbags = 3
    input_size = len(feat_num) + 2 * len(feat_cat)
    preds = np.zeros(yte.shape)

    for j in range(nbags):
        print('  - bag %d' %(j+1))
        net = nn1(input_size=input_size, hidden_size=25, output_size=199, batch_size=128, n_unique=n_unique).to(device)
        net.train_epochs(nepochs=60, X=xtr, y=ytr)
        preds += net.predict(xte).cpu().detach().numpy()

    preds /= nbags
    loss = np.mean((yte.cpu().detach().numpy()-preds)**2)
    print('  - CRPS=%.5f' % (loss))

    observed.append(yte.cpu().detach().numpy())
    predicted.append(preds)
    playids.append(playid_te)


print('- compute overall CV score')
observed = np.concatenate(observed)
predicted = np.concatenate(predicted)
playids = np.concatenate(playids)

loss = np.mean((observed-predicted)**2)
print('- CRPS=%.5f' %(loss))

print('Done!!!')


# - Fold 1
#   - clean data
#   - get features
#   - transform features
#   - train network
#   - bag 1
#   - bag 2
#   - bag 3
#   - CRPS=0.01353
# - Fold 2
#   - clean data
#   - get features
#   - transform features
#   - train network
#   - bag 1
#   - bag 2
#   - bag 3
#   - CRPS=0.01422
# - Fold 3
#   - clean data
#   - get features
#   - transform features
#   - train network
#   - bag 1
#   - bag 2
#   - bag 3
#   - CRPS=0.01340
# - Fold 4
#   - clean data
#   - get features
#   - transform features
#   - train network
#   - bag 1
#   - bag 2
#   - bag 3
#   - CRPS=0.01415
# - Fold 5
#   - clean data
#   - get features
#   - transform features
#   - train network
#   - bag 1
#   - bag 2
#   - bag 3
#   - CRPS=0.01413
# - compute overall CV score
# - CRPS=0.01387
# Done!!!




# - read data
# - merge cv folds to train data
# - Fold 1
#   - clean data
#   - get features
#   - transform features
#   - train network
#   - bag 1
#   - bag 2
#   - bag 3
#   - CRPS=0.01316
# - Fold 2
#   - clean data
#   - get features
#   - transform features
#   - train network
#   - bag 1
#   - bag 2
#   - bag 3
#   - CRPS=0.01361
# - Fold 3
#   - clean data
#   - get features
#   - transform features
#   - train network
#   - bag 1
#   - bag 2
#   - bag 3
#   - CRPS=0.01287
# - Fold 4
#   - clean data
#   - get features
#   - transform features
#   - train network
#   - bag 1
#   - bag 2
#   - bag 3
#   - CRPS=0.01312
# - Fold 5
#   - clean data
#   - get features
#   - transform features
#   - train network
#   - bag 1
#   - bag 2
#   - bag 3
#   - CRPS=0.01301
# - compute overall CV score
# - CRPS=0.01316
# Done!!!
