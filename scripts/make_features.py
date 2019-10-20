"""
Feature Generation Function(s)
1. Each function will define a new feature, a scaler per PLAY
2. Can assumed given clean data (see clean_data.py)
3. Final function calls all previous functions to generate new columns
4. In the end, will have one row per PLAY

NOTE:
    All (finished) feature functions need to start with "f_" and will be called in
    ALPHABETICAL order. If make features using the features...well, it gets
    more complicated. Probably need a second order "new_features_2" that calls
    on the produced features etc.
NOTE:
    You can provide list of features to make_features, these are the feature names
    and need to match the functions. The "f_" will be appended in the function to
    make creating the feature list nicer/simpler. Naming will be important still.
"""

import pandas as pd
from scripts.clean_data import DataCleaner


class FeatureGenerator:
    def __init__(self, df):
        self.response = ["Yards"]
        self.time_features = ['TimeHandoff', 'TimeSnap']
        # repeated features; PlayId for index; Yards for response
        self.repeated_features = ['PlayId', 'YardLine', 'Quarter', 'PossessionTeam', 'Down',
                                  'Distance', 'OffenseFormation', 'OffensePersonnel',
                                  'DefendersInTheBox', 'DefensePersonnel', 'HomeTeamAbbr',
                                  'VisitorTeamAbbr', 'Week', 'StadiumType', 'Turf', 'GameWeather']

    def yardsTillNow(self, df):
        # will we be able to do this on the test data?
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

    def distanceToGoal(self, df):
        # should replace YardLine
        pass

    def f_test(self, df):
        return 1

    def new_features(self, df, features):
        if features is None:
            methods = [method for method in dir(self)
                       if callable(getattr(self, method)) if method.startswith('f_')]
        else:
            methods = ["f_" + feature for feature in features]
        out = {}
        for method in methods:
            out[method[2:]] = getattr(self, method)(df)
        return pd.Series(out)

    def make_features(self, df, is_train, features=None):
        # select features we want that are the same for each
        out = df[self.repeated_features].drop_duplicates()
        # compute new features based on data.frame
        extractedFeatures = df.groupby('PlayId').apply(self.new_features, features)
        x = out.set_index('PlayId').join(extractedFeatures).copy()
        y = df[['PlayId']+self.response].drop_duplicates().set_index('PlayId') if is_train else []
        return x, y, out.index.values


if __name__ == "__main__":
    data = pd.read_csv('../input/train.csv', low_memory=False)
    cleaner = DataCleaner(data)
    data = cleaner.clean_data(data)
    ctor = FeatureGenerator(data)  # feature constructor
    x, y, playid = ctor.make_features(data, ["test"])
    x.head()
    x, y, playid = ctor.make_features(data)

