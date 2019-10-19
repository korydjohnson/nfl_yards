"""
Feature Generation Function(s)
1. Each function will define a new feature, a scaler per PLAY
2. Can assumed given clean data (see clean_data.py)
3. Final function calls all previous functions to generate new columns
4. In the end, will have one row per PLAY

NOTE:
    All feature functions need to start with "f_" and will be called in
    ALPHABETICAL order. If make features using the features...well, it gets
    more complicated. Probably need a second order "new_features_2" that calls
    on the produced features etc.
"""

import pandas as pd
from clean_data import DataCleaner


class FeatureGenerator:
    def __init__(self, df):
        self.response = ["Yards"]
        self.time_features = ['TimeHandoff', 'TimeSnap']
        # repeated features that we want to keep in X; PlayId for index
        self.repeated_features = ['PlayId', 'YardLine', 'Quarter', 'PossessionTeam', 'Down',
                                  'Distance', 'OffenseFormation', 'OffensePersonnel', 'Yards',
                                  'DefendersInTheBox', 'DefensePersonnel', 'HomeTeamAbbr',
                                  'VisitorTeamAbbr', 'Week', 'StadiumType', 'Turf', 'GameWeather']

    def f_yardsTillNow(self, df):
        pass

    def f_snow(self, df):
        pass

    def f_distanceToOpposingTeam(self, df):
        pass

    def f_isOpening(self, df):
        pass

    def f_openingSize(self, df):
        pass

    def f_specialYardIndicators(self, df):
        # First and ten/15/20
        pass

    def f_shortRunSetup(self, df):
        # everyone is on line of scrimmage, see plots
        pass

    def f_timeTillHandoff(self, df):
        pass

    def f_positionOfRunner(self, df):
        pass

    def f_distanceToGoal(self, df):
        # should replace YardLine
        pass

    def new_features(self, df):
        out = {}
        methods = [method for method in dir(self)
                   if callable(getattr(self, method)) if method.startswith('f_')]
        for method in methods:
            out[method[2:]] = getattr(self, method)(df)
        return pd.Series(out)

    def make_features(self, df):
        # select features we want that are the same for each
        out = df[self.repeated_features].drop_duplicates()
        # compute new features based on data.frame
        extractedFeatures = df.groupby('PlayId').apply(self.new_features)
        out = out.set_index('PlayId').join(extractedFeatures)
        x = out.drop(columns=self.response)
        y = out[self.response]
        return x, y, out.index.values


if __name__ == "__main__":
    data = pd.read_csv('../input/train.csv', low_memory=False)
    cleaner = DataCleaner(data)
    data = cleaner.clean_data(data)
    # data["PlayId2"] = data["PlayId"]  # just for testing; check group
    ctor = FeatureGenerator(data)  # feature constructor
    x, y, playid = ctor.make_features(data)

