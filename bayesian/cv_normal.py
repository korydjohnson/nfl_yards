
import numpy as np
import pandas as pd
import scipy.optimize as optimize
from scipy.stats import norm

from eda_Py.clean_data import DataCleaner
from utils.foptims import f1
from utils.fgen import FeatGenerator

print('- read data')
df = pd.read_csv('./input/train.csv', low_memory=False)
folds = pd.read_csv('./input/cvfolds.csv')

print('- merge cv folds to train data')
df = df.merge(folds[['PlayId', 'fold']], on = 'PlayId', how = 'left')

fgen = FeatGenerator()
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
    xtr, ytr, playid_tr = fgen.get_features(df_tr)
    xte, yte, playid_te = fgen.get_features(df_te)

    print('  - optimize objective')
    val = np.log(df_tr['Yards'].clip(-14, 99) + 15)
    init_params = [val.mean(), val.std()]
    result = optimize.minimize(f1, init_params, args=[ytr])

    print('  - get predictions')
    p1, p2 = result.x
    vals = np.arange(-99, 100)
    vals_scaled = np.log(vals.clip(-14, 99) + 15)
    preds = norm.cdf(vals_scaled, p1, p2)
    preds = np.tile(preds, (len(yte), 1))

    loss = np.mean((yte-preds)**2)
    print('  - CRPS=%.5f' % (loss))

    observed.append(yte)
    predicted.append(preds)
    playids.append(playid_te)


print('- compute overall CV score')
observed = np.concatenate(observed)
predicted = np.concatenate(predicted)
playids = np.concatenate(playids)

loss = np.mean((observed-predicted)**2)
print('- CRPS=%.5f' %(loss))

print('Done!!!')
