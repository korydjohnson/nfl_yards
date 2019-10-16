
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from network.network import nn1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('- Read data.')
train = pd.read_csv('./input/featureset.csv')


print('- Split train/test')
ntr = 18000
tr = train[:ntr].reset_index(drop=True)
te = train[ntr:].reset_index(drop=True)

print('- create target')
ytr = pd.get_dummies(pd.Categorical(tr['Yards']+99, range(199))).cumsum(axis=1).values.astype(float)
yte = pd.get_dummies(pd.Categorical(te['Yards']+99, range(199))).cumsum(axis=1).values.astype(float)

ytr = torch.from_numpy(ytr).float().to(device)
yte = torch.from_numpy(yte).float().to(device)


print('- Process numerical features')
feat_num = [
  'YardLine', 'HomeScoreBeforePlay', 'VisitorScoreBeforePlay', 'Distance', 'Down',
  'DefendersInTheBox_div_Distance', 'PossessionHome', 'Temperature', 'WindSpeed',
]

scaler = StandardScaler()
scaler.fit(tr[feat_num])
tr[feat_num] = scaler.transform(tr[feat_num])
te[feat_num] = scaler.transform(te[feat_num])

xtr_num = tr[feat_num].values.astype(float)
xte_num = te[feat_num].values.astype(float)

xtr_num = torch.from_numpy(xtr_num).float().to(device)
xte_num = torch.from_numpy(xte_num).float().to(device)


print('- Process categorical features')
feat_cat = [
  'HomeTeamAbbr', 'VisitorTeamAbbr', 'OffenseFormation',
]

n_unique = tr[feat_cat].max().values + 1

xtr_cat = tr[feat_cat].values.astype(float)
xte_cat = te[feat_cat].values.astype(float)

xtr_cat = torch.from_numpy(xtr_cat).long().to(device)
xte_cat = torch.from_numpy(xte_cat).long().to(device)


print('- Initialize network')
input_size = len(feat_num) + 2*len(feat_cat)
net = nn1(input_size=input_size, hidden_size=50, output_size=199, batch_size=512, n_unique=n_unique).to(device)


print('- Train network')
xtr = [xtr_num, xtr_cat]
xte = [xte_num, xte_cat]

for i in range(100):
  net.train_epochs(nepochs=1, X=xtr, y=ytr)
  loss = net.eval_model(xte, yte)
  print(loss)



