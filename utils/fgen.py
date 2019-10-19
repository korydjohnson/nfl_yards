
import pandas as pd


class FeatGenerator:

    def __init__(self):
        self.base_feat = ['DefendersInTheBox', 'Distance']
        self.target = 'Yards'

    def get_features(self, df):
        cols = self.base_feat + [self.target]
        df = df.groupby('PlayId')[cols].first().reset_index()

        x = df[self.base_feat].values
        y = pd.get_dummies(pd.Categorical(df[self.target] + 99, range(199))).cumsum(axis=1).values
        playid = df['PlayId'].values

        return x, y, playid