"""
Feature Generation Function(s)
1. Each function will define a new feature, a scaler per PLAY
2. Can assumed given clean data (see clean_data.py)
3. Final function calls all previous functions to generate new columns
4. In the end, will have one row per PLAY

NOTE:
    All feature functions need to start with "feature" and will be called in
    ALPHABETICAL order. If make features using the features...well, it gets
    more complicated. Probably need a second order "new_features_2" that calls
    on the produced features etc.
"""

import pandas as pd
from clean_data import DataCleaner


class FeatureGenerator:
    def __init__(self, df):
        self.repeated_features = ['GameId', 'PlayId', 'YardLine', 'Quarter', 'PossessionTeam',
                                  'Down', 'Distance', 'FieldPosition', 'OffenseFormation',
                                  'OffensePersonnel', 'DefendersInTheBox', 'DefensePersonnel',
                                  'PlayDirection', 'TimeHandoff', 'TimeSnap', 'Yards',
                                  'HomeTeamAbbr', 'VisitorTeamAbbr', 'Week', 'StadiumType', 'Turf',
                                  'GameWeather']
        self.unique_features = [col for col in df.columns if col not in self.repeated_features]

    def feature_a(self, play_df):
        return play_df["PlayId2"].unique()

    def feature_b(self, play_df):
        return play_df["PlayId2"].unique()

    # def feature_isRusher(self, df):
    #     # Create a variable if the player is the rusher
    #     df["Is_rusher"] = df["NflId"] == df["NflIdRusher"]

    def new_features(self, df):
        out = {}
        methods = [method for method in dir(self)
                   if callable(getattr(self, method)) if method.startswith('feature')]
        for method in methods:
            out[method[8:]] = getattr(self, method)(df)
        return pd.Series(out)

    def make_features(self, df):
        # select features we want that are the same for each
        out = df[self.repeated_features].drop_duplicates()
        # compute new features based on data.frame
        extractedFeatures = df.groupby('PlayId').apply(self.new_features)
        out = out.set_index('PlayId').join(extractedFeatures)
        return out


if __name__ == "__main__":
    data = pd.read_csv('../input/train.csv', low_memory=False)
    cleaner = DataCleaner(data)
    data = cleaner.clean_data(data)
    data["PlayId2"] = data["PlayId"]  # just for testing; check group
    ctor = FeatureGenerator(data)  # feature constructor
    features = ctor.make_features(data)
    features.head()
