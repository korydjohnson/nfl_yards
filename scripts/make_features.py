"""
Feature Generation Function(s)
1. Each function will define a new feature, a scaler per PLAY
2. Can assumed given clean data (see clean_data.py)
3. Final function calls all previous functions to generate new columns
4. In the end, will have one row per PLAY
5. Functions take entire play information and input is labelled dfP.

NOTE:
    All (finished) feature functions need to start with "f_" and will be called in
    ALPHABETICAL order. If make features using the features, need to put dependence
    in name as "newFeatures_dependsOn". Where "dependsOn" function needs to come first.
    While accurate naming isn't necessary, including "_" is.
NOTE:
    You can provide list of features to make_features, these are the feature names
    and need to match the functions. The "f_" will be appended in the function to
    make creating the feature list nicer/simpler. Naming will be important still.
"""

import pandas as pd
import numpy as np
from scripts.clean_data import DataCleaner
# from clean_data import DataCleaner


class FeatureGenerator:
    def __init__(self):
        self.response = ["Yards"]
        self.time_features = ['TimeHandoff', 'TimeSnap']
        # repeated features; PlayId excluded as it's the index; Yards for response
        self.repeated_features = ['Quarter', 'PossessionTeam', 'Down',
                                  'Distance', 'OffenseFormation', 'OffensePersonnel',
                                  'DefendersInTheBox', 'DefensePersonnel', 'HomeTeamAbbr',
                                  'VisitorTeamAbbr', 'Week', 'StadiumType', 'Turf', 'GameWeather',
                                  'DistanceToGoal', 'LineOfScrimmage']
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
    def f_RusherInfo(dfP):
        bool_rusher = dfP['NflId'] == dfP['NflIdRusher']
        s = dfP[bool_rusher]
        mates = dfP[(dfP.Team == s.Team.values[0]) & ~bool_rusher]
        opponents = dfP[dfP.Team != s.Team.values[0]]
        dist_mates =  np.sqrt((mates.X - s['X'].values[0])**2 + (mates.Y - s['Y'].values[0])**2)
        dist_opponents = np.sqrt((opponents.X - s['X'].values[0])**2 + (opponents.Y - s['Y'].values[0])**2)
        closest_opponent = opponents.loc[dist_opponents.idxmin(),:]

        AccClosestvsRusher = (closest_opponent['A']-s['A']).values[0]
        SpeedClosestvsRusher = (closest_opponent['S']-s['S']).values[0]
        DistDefvsOff = dist_opponents.sum() - dist_mates.sum()
        DistOffMean = dist_mates.mean()
        DistDefMean = dist_opponents.mean()
        DistDef = dist_opponents.min()

        DistLOS = (s['LineOfScrimmage'] - s['X']).values[0] * \
            np.where(s['PlayDirection'] == 'right', 1, -1)[0]
        DistGoal = \
            110-s['X'].values[0] if s['PlayDirection'].values[0] == 'right' else s['X'].values[0]
        Acc = s['A'].values[0]
        radian_angle = (90 - s['Dir']) * np.pi / 180.0
        SpeedX = np.abs(s['S'] * np.cos(radian_angle)).values[0]
        radian_angle = (90 - s['Dir']) * np.pi / 180.0
        SpeedY = np.abs(s['S'] * np.sin(radian_angle)).values[0]
        Pos = s.Position.values[0]
        d = {"DistLOS": DistLOS, "DistGoal": DistGoal, "DistDef": DistDef,
             "Acc": Acc, "SpeedX": SpeedX, "SpeedY": SpeedY, "Pos": Pos,
             "DistDefvsOff": DistDefvsOff, "DistOffMean": DistOffMean,
             "DistDefMean": DistDefMean, "AccClosestvsRusher": AccClosestvsRusher,
             "SpeedClosestvsRusher": SpeedClosestvsRusher}
        return d

    @staticmethod
    def f_DistanceToLOS(dfP):
        return (dfP.X - dfP.LineOfScrimmage).abs().mean().__float__()

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

        # call methods in correct order
        maxOrder = max([method.count("_") for method in methods])
        for order in range(maxOrder):
            methods_sub = [method for method in methods if method.count("_") == order+1]
            extractedFeatures = df.groupby('PlayId').apply(self.new_features, methods_sub)
            df = df.join(extractedFeatures)

        out = df.drop_duplicates(self.repeated_features)

        # return based on training or test data
        if test:
            covariates = out.drop(columns=self.dropColumns)
            return covariates, out.index.values
        else:
            covariates = out.drop(columns=self.response).drop(columns=self.dropColumns)
            return covariates, out[self.response], out.index.values


if __name__ == "__main__":
    data = pd.read_csv('../input/train.csv', low_memory=False)
    cleaner = DataCleaner(data)
    data = cleaner.clean_data(data)
    ctor = FeatureGenerator()  # feature constructor
    dfSub = data.filter(like="20170907000118", axis=0)
    x, y, PlayId = ctor.make_features(dfSub)
    x
    dfSub = dfSub.drop("Yards", axis=1)
    x, PlayId = ctor.make_features(dfSub, test=True)
    x
    x, y, PlayId = ctor.make_features(data, ["DistanceToLOS"])
    x.head()
    x, y, PlayId = ctor.make_features(data)
    x.head()
    x.to_csv("../input/trainClean_py.csv")
    y.to_csv("../input/trainResponse_py.csv")
