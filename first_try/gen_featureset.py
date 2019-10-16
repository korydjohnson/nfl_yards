
import numpy as np
import pandas as pd

dtypes = {'GameId': 'object', 'PlayId': 'object', 'NflId': 'object', 'NflIdRusher': 'object'}
date_cols = ['GameClock', 'TimeHandoff', 'TimeSnap', 'PlayerBirthDate']

df = pd.read_csv('./input/trainClean.csv', dtype=dtypes, parse_dates=date_cols, low_memory=False)

#### new features
df['DefendersInTheBox_div_Distance'] = df['DefendersInTheBox'] / df['Distance']
df['PossessionHome'] = np.where(df['PossessionTeam']==df['HomeTeamAbbr'], 1, 0)
#df['MintuesToPlay'] = df['GameClock'].dt.hour

#print(df.to_string())

#### columns that are identical for each PlayId
cols = [
    ## numerical cols
    'YardLine', 'HomeScoreBeforePlay', 'VisitorScoreBeforePlay', 'Distance', 'Down',
    'DefendersInTheBox_div_Distance', 'PossessionHome', 'Temperature', 'WindSpeed',
    ## categorical columns
    'HomeTeamAbbr', 'VisitorTeamAbbr', 'OffenseFormation',
    ## target
    'Yards',
]

train = df.groupby('PlayId')[cols].first().reset_index()

#### replace missing values
train = train.fillna(train.mean())

#### categories to integer
train['HomeTeamAbbr'] = pd.Categorical(train['HomeTeamAbbr']).codes
train['VisitorTeamAbbr'] = pd.Categorical(train['VisitorTeamAbbr']).codes
train['OffenseFormation'] = pd.Categorical(train['OffenseFormation']).codes

train.to_csv('./input/featureset.csv', index=False)


