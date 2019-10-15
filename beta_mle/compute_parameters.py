
import numpy as np
import pandas as pd
from scipy.stats import beta
import scipy.optimize as optimize

dtypes = {'GameId': 'object', 'PlayId': 'object', 'NflId': 'object', 'NflIdRusher': 'object'}
date_cols = ['GameClock', 'TimeHandoff', 'TimeSnap', 'PlayerBirthDate']

df = pd.read_csv('./input/trainClean.csv', dtype=dtypes, parse_dates=date_cols)

cols = ['YardLine', 'HomeScoreBeforePlay', 'VisitorScoreBeforePlay', 'Distance', 'Down', 'Yards']
a = df.groupby('PlayId')[cols].mean().reset_index()


#### split train/test
ntr = 18000
tr = a[:ntr].reset_index(drop=True)
te = a[ntr:].reset_index(drop=True)


#### mle
x = (tr['Yards'].values + 100) / 200

def f(params):
    p1, p2 = params
    score = -beta.logpdf(x, a=p1, b=p2).mean()
    return score


init_params = [100,100]
result = optimize.minimize(f, init_params)

print(result.x)

p1, p2 = result.x
probs = beta.cdf(np.arange(199)/198, p1, p2)
probs = np.tile(probs, (len(te), 1))

o = te['Yards'].values
obs = np.zeros((len(te),199))
obs[range(len(te)),o+99] = 1
obs = obs.cumsum(axis = 1)

score = np.mean((obs-probs)**2)
print(score)


#### optimize metric directly
o = tr['Yards'].values
obs = np.zeros((len(tr),199))
obs[range(len(tr)),o+99] = 1
obs = obs.cumsum(axis = 1)

def f(params):
    p1, p2 = params
    prob = beta.cdf((np.arange(199)+1)/200, p1, p2)
    prob = np.tile(prob, (len(obs), 1))
    score = np.mean((obs - prob) ** 2)
    return score

init_params = [100,100]
result = optimize.minimize(f, init_params)

print(result.x)

p1, p2 = result.x
probs = beta.cdf(np.arange(199)/198, p1, p2)
probs = np.tile(probs, (len(te), 1))

o = a.loc[ntr:,'Yards'].values
obs = np.zeros((len(o),199))
obs[list(range(len(o))),o+99] = 1
obs = obs.cumsum(axis = 1)

score = np.mean((obs-probs)**2)
print(score)