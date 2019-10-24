"""
Feature Generation Function(s)
1. Each function will define a new feature, a scaler per PLAY
2. Can assumed given clean data (see clean_data.py)
3. Final function calls all previous functions to generate new columns
4. In the end, will have one row per PLAY
5. Functions take entire play information and input is labelled dfP.

NOTE:
    All (finished) feature functions need to start with "f_" and will be called in
    ALPHABETICAL order. Since we are unnesting a dictionary, it is easy to have functions
    call other functions. These secondary functions cannot start with "f_".
NOTE:
    You can provide list of features to make_features, these are the feature names
    and need to match the functions. The "f_" will be appended in the function to
    make creating the feature list nicer/simpler.
"""

import pandas as pd
import numpy as np
# from scripts.clean_data_rescale import DataCleaner
# from clean_data_rescale import DataCleaner


class FeatureGenerator:
    def __init__(self):
        self.response = ["Yards"]
        self.time_features = ['TimeHandoff', 'TimeSnap']
        # repeated features; PlayId excluded as it's the index; Yards for response
        self.repeated_features = ['Quarter', 'PossessionTeam', 'Down',
                                  'Distance', 'OffenseFormation', 'OffensePersonnel',
                                  'DefendersInTheBox', 'DefensePersonnel', 'HomeTeamAbbr',
                                  'VisitorTeamAbbr', 'Week', 'StadiumType', 'Turf', 'GameWeather',
                                  'LineOfScrimmage']
        self.dropColumns = ["PlayDirection"]

    def yardsTillNow(self, df):
        # will we be able to do this on the test data?
        # if get runs in order, then could sum as we see them
        pass

    def snow(self, df):
        pass

    def isOpening(self, df):
        pass

    def openingSize(self, df):
        pass

    def specialYardIndicators(self, df):
        # First and ten/15/20
        pass

    def timeTillHandoff(self, df):
        pass

    def positionOfRunner(self, df):
        pass

    @staticmethod
    def f_TeamInfo(dfP):
        offense = dfP[dfP.OnOffense]
        defense = dfP[~dfP.OnOffense]
        DefXStd = defense['X'].std()
        DefYStd = defense['Y'].std()
        OffXStd = offense['X'].std()
        OffYStd = offense['Y'].std()
        OffCountWR = offense['Position'].isin(['WR']).sum()
        DefDistLOS = (defense.X - defense.LineOfScrimmage).abs().mean().__float__()
        OffDistLOS = (offense.X - offense.LineOfScrimmage).abs().mean().__float__()
        d = {"DefXStd": DefXStd, "DefYStd": DefYStd, "OffXStd": OffXStd, "OffYStd": OffYStd,
             "OffCountWR": OffCountWR, "DefDistLOS": DefDistLOS, "OffDistLOS": OffDistLOS}
        return d

    @staticmethod
    def f_RusherInfo(dfP):
        # set-up
        s = dfP[dfP['NflId'] == dfP['NflIdRusher']]
        rush_dir_rad = (90 - s['Dir']) * np.pi / 180.0
        offense = dfP[dfP.OnOffense]
        defense = dfP[~dfP.OnOffense]
        dist_off = np.sqrt((offense.X - s['X'].values[0])**2 + (offense.Y - s['Y'].values[0])**2)
        dist_def = np.sqrt((defense.X - s['X'].values[0])**2 + (defense.Y - s['Y'].values[0])**2)
        closest_opponent = defense.loc[dist_def.idxmin(), :]

        # descriptive statistics
        ADef = (closest_opponent['A']-s['A']).values[0]
        SDef = (closest_opponent['S']-s['S']).values[0]
        DistDef = dist_def.min()
        Acc = s['A'].values[0]
        SpeedX = np.abs(s['S'] * np.cos(rush_dir_rad)).values[0]
        SpeedY = np.abs(s['S'] * np.sin(rush_dir_rad)).values[0]
        Pos = s.Position.values[0]

        # interpretive/computed statistics
        DistOffMean = dist_off.mean() * 11 / 10  # rescale for rusher 0
        DistDefMean = dist_def.mean()
        DistLOS = (s['LineOfScrimmage'] - s['X']).values[0]

        # saving results
        d = {"DistLOS": DistLOS, "DistDef": DistDef, "Acc": Acc,
             "SpeedX": SpeedX, "SpeedY": SpeedY, "Pos": Pos,
             "DistOffMean": DistOffMean, "DistDefMean": DistDefMean,
             "ADef": ADef, "SDef": SDef}
        return d

    def new_features(self, dfP, methods):
        out = {}
        for method in methods:
            out[method[2:]] = getattr(self, method)(dfP)  # always store features as dict
        out = pd.io.json.json_normalize(out, sep='_')
        return out.iloc[0]

    def make_features(self, df, features=None, test=False):
        # creating features, method names
        if features is None:
            methods = [method for method in dir(self)
                       if callable(getattr(self, method)) if method.startswith('f_')]
        else:
            methods = ["f_" + feature for feature in features]

        out = df.drop_duplicates(self.repeated_features)
        extractedFeatures = df.groupby('PlayId').apply(self.new_features, methods)
        out = out.join(extractedFeatures)

        # return based on training or test data
        if test:
            covariates = out.drop(columns=self.dropColumns)
            return covariates, out.index.values
        else:
            covariates = out.drop(columns=self.response).drop(columns=self.dropColumns)
            return covariates, out[self.response], out.index.values


if __name__ == "__main__":
    data = pd.read_csv('../input/trainClean_py.csv', low_memory=False).set_index("PlayId")
    ctor = FeatureGenerator()  # feature constructor
    dfSub = data.filter(like="20170907000118", axis=0)
    x, y, PlayId = ctor.make_features(dfSub)
    x
    x, y, PlayId = ctor.make_features(dfSub, ["DistanceToLOS"])
    x
    dfSub = dfSub.drop("Yards", axis=1)
    x, PlayId = ctor.make_features(dfSub, test=True)
    x
    x, PlayId = ctor.make_features(dfSub, features=["DistanceToLOS"], test=True)
    x
    x, y, PlayId = ctor.make_features(data)
    x.head()
    for c in x.columns:
        print(x[c].sample(10))
    x.to_csv("../input/features_py.csv")
    y.to_csv("../input/response_py.csv")
