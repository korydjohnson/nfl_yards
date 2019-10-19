"""
Creating function to clean training and test data for nfl_yards

This is more the "piece things together" file; so random code chunks etc

Contains older code, more copy paste from others, not final

1. Training data can be cleaned all-together
2. Test data will be received as df of play information for each play
3. Outer function "prepData" will call two functions "cleanData" and "getFeatures"
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

data = pd.read_csv('../input/train.csv')

"""
Cleaning Function

1. Features not cleaned because deemed a-priori irrelevant
2. Don't expect to be relevant: "HomeScoreBeforePlay", "VisitorScoreBeforePlay"
3. Compress categories: StadiumType, Turf, Weather,
"""


# Missing data
data.isnull().sum(axis=0)
missDir = data.Dir.isnull()
# all missing values are from strong safeties...I'm going to assume they aren't moving
data.Position[missDir]
missOr = data.Orientation.isnull()
# tight ends and strong safeties
data.Position[missOr]
data[["PlayId", "Position", "Dir", "Orientation"]][missOr]


import matplotlib.pyplot
# why are there more around 0 and 180? figured it'd be around 90 and 270... lots of people
# are looking across the field, not down it.
matplotlib.pyplot.hist(data.Dir)

"""
helper objects for cleaning Pipeline
"""

# Columns to Delete: stadium, location,
drop_columns = ["GameClock", "DisplayName", "JerseyNumber", "NflId", "Season",
                "NflIdRusher", "TimeHandoff", "TimeSnap", "PlayerBirthDate",
                "PlayerCollegeName", "Stadium", "Location", "WindSpeed", "WindDirection",
                "HomeScoreBeforePlay", "VisitorScoreBeforePlay"]

# fixing abbreviations
mapping_team_dict = {"ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU"}

# StadiumType
outdoor = ['Outdoor', 'Outdoors', 'Cloudy', 'Heinz Field', 'Outdor',
           'Ourdoor', 'Outside', 'Outddors', 'Outdoor Retr Roof-Open', 'Oudoor', 'Bowl']
indoorClosed = ['Indoors', 'Indoor', 'Indoor, Roof Closed', 'Indoor, Roof Closed',
                'Retractable Roof', 'Retr. Roof-Closed', 'Retr. Roof - Closed',
                'Retr. Roof Closed']
indoorOpen = ['Indoor, Open Roof', 'Open', 'Retr. Roof-Open', 'Retr. Roof - Open']
domeClosed = ['Dome', 'Domed, closed', 'Closed Dome', 'Domed', 'Dome, closed']
domeOpen = ['Domed, Open', 'Domed, open']
newNames = ["outdoor", "indoorClosed", "indoorOpen", "domeClosed", "domeOpen"]
nameList = [outdoor, indoorClosed, indoorOpen, domeClosed, domeOpen]
stadium_type_map = {nameList[i][j]: newNames[i] for i in range(len(newNames))
                    for j in range(len(nameList[i]))}

"""
All functions need a "train" option that sets encoders etc, or test which uses them
--> need to save encoders to global...
Could actually just test if given only a single PlayId
Need a default value in case it is missing/not included in previous set
hmmm, just get rid of encoders should do the trick... will be categorical anyway
"""


def label_encoder_teams(df, test=False):  # Encode the team values
    columns = ["PossessionTeam", "FieldPosition", "HomeTeamAbbr", "VisitorTeamAbbr"]

    for col in columns:
        df.fillna({col: "Other"}, inplace=True)  # fill missing
        df[col].map(mapping_team_dict)  # clean names

    if not test:
        le_teams = LabelEncoder()
        unique_features = list(
            set(list(df["PossessionTeam"].unique()) + list(df["FieldPosition"].unique()) \
                + list(df["HomeTeamAbbr"].unique()) + list(df["VisitorTeamAbbr"].unique())))
        le_teams.fit(unique_features)
    else:


    for col in columns:
        df[col] = pd.Categorical(le_teams.transform(df[col].values))


def label_encoder_team_formation(df):  # Encode team formation
    columns = ["OffenseFormation", "OffensePersonnel", "DefensePersonnel"]
    for col in columns:
        df.fillna({col: "Other"}, inplace=True)
        le = LabelEncoder()
        df[col] = pd.Categorical(le.fit_transform(df[col].values))


def label_encoder_stadium(df):  # Encode stadium type

    df.fillna({"Stadium": "Other"}, inplace=True)
    le = LabelEncoder()
    df["Stadium"] = le.fit_transform(df["Stadium"].values)


def input_defenders(df):
    si = SimpleImputer("most_frequent")
    df["DefendersInTheBox"] = si.fit_transform(df["DefendersInTheBox"].values).reshape(-1, 1)


def convert_height(df):
    temp = df.PlayerHeight.str.split("-", expand=True)
    df.PlayerHeight = pd.to_numeric(temp[0]) * 12 + pd.to_numeric(temp[1])
    df.PlayerHeight.head()


# Data Process Pipeline
def clean_data(df):
    # Create a variable if the player is the rusher
    df["Is_rusher"] = df["NflId"] == df["NflIdRusher"]

    # modify in place
    label_encoder_teams(df)

    # encode data away or home
    df["Team"] = df["Team"].apply(lambda x: 0 if x == "away" else 1)

    # input the nan from defenders in the box with the most common value
    input_defenders(df)

    # encode the teams formation and offence strategy
    label_encoder_team_formation(df)

    # clean stadium types
    df.fillna({"StadiumType": "Other"}, inplace=True)
    df["StadiumType"] = df["StadiumType"].map(stadium_type_map, na_action='ignore')
    df["StadiumType"] = df["StadiumType"].apply(lambda x: 1 if x == "Outdoor" else 0)

    # play direction
    df["PlayDirection"] = df["PlayDirection"].map({"left": 1, "right": 0}, na_action="ignore")

    # sort and drop irrelevant columns
    df.sort_values(by=["GameId", "PlayId"]).reset_index()
    df.drop(drop_columns, inplace=True, axis=1)

    return df


processed_df = clean_data(data)
