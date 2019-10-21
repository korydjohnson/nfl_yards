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
from clean_data import DataCleaner


class FeatureGenerator:
    def __init__(self):
        self.response = ["Yards"]
        self.time_features = ['TimeHandoff', 'TimeSnap']
        # repeated features; PlayId excluded as it's the index; Yards for response
        self.repeated_features = ['YardLine', 'Quarter', 'PossessionTeam', 'Down',
                                  'Distance', 'OffenseFormation', 'OffensePersonnel', 'Yards',
                                  'DefendersInTheBox', 'DefensePersonnel', 'HomeTeamAbbr',
                                  'VisitorTeamAbbr', 'Week', 'StadiumType', 'Turf', 'GameWeather']
        self.dropColumns = ["PlayDirection"]

    def yardsTillNow(self, df):
        # will we be able to do this on the test data?
        # if get runs in order, then could sum as we see them
        pass

    def snow(self, df):
        pass

    def distanceToOpposingTeam(self, df):
        pass

    def isOpening(self, df):
        pass

    def openingSize(self, df):
        pass

    def specialYardIndicators(self, df):
        # First and ten/15/20
        pass

    def shortRunSetup(self, df):
        # everyone is on line of scrimmage, see plots
        pass

    def timeTillHandoff(self, df):
        pass

    def positionOfRunner(self, df):
        pass

    def f_DistanceToGoal(self, dfP):  # should replace YardLine
        dfR = dfP.drop_duplicates(self.repeated_features)
        out = dfR.YardLine + (100-2*dfR.YardLine)*(dfR.PossessionTeam == dfR.FieldPosition)
        return out.__int__()

    def f_DistanceToLOS(self, dfP):
        dfR = dfP.drop_duplicates(self.repeated_features)
        condition = (dfR.PlayDirection == "right") & (dfR.PossessionTeam == dfR.FieldPosition)
        los_side = "left" if condition.values else "right"
        los = (dfR.YardLine + 10 if los_side == "left" else 110 - dfR.YardLine).__int__()
        return (dfP.X - los).abs().mean().__float__()

    def new_features(self, df, methods):
        out = {}
        for method in methods:
            out[method[2:]] = getattr(self, method)(df)
        return pd.Series(out)

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
        covariates = out.drop(columns=self.response).drop(columns=self.dropColumns)

        # return based on training or test data
        if test:
            return covariates, out.index.values
        else:
            return covariates, out[self.response], out.index.values


if __name__ == "__main__":
    data = pd.read_csv('../input/train.csv', low_memory=False)
    cleaner = DataCleaner(data)
    data = cleaner.clean_data(data)
    ctor = FeatureGenerator()  # feature constructor
    x, y, PlayId = ctor.make_features(data, ["DistanceToGoal"])
    x.head()
    dfSub = data.filter(like="20170907000118", axis=0)
    dfSub
    x, y, PlayId = ctor.make_features(dfSub)
    x
    x, PlayId = ctor.make_features(dfSub, test=True)
    x
    x, y, PlayId = ctor.make_features(data)
    x.head()
