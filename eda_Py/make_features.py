"""
Feature Generation Function(s)
1. Each function will define a new feature
2. Can assumed given clean data (see clean_data.py)
3. Final function calls all previous functions to generate new columns
"""

import pandas as pd
from clean_data import clean_data


def make_features(df):
    return df


if __name__ == "__main__":
    dfClean = clean_data(pd.read_csv('../input/train.csv', low_memory=False))
    make_features(dfClean)
