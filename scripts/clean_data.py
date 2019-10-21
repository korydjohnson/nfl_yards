"""
Creating function to clean training and test data for nfl_yards
1. Training data can be cleaned all-together
2. Test data will be received as df of play information for each play
3. Require suitable globals to be used on train/test
4. Need to ensure that don't get errors if get category we haven't seen before.
   True both for cleaning and for models used.
5. Have to have all team abbreviations with same category numbers (check equivalence for features).

Cleaning Function
1. Features not cleaned because deemed a-priori irrelevant
2. Don't expect to be relevant: "HomeScoreBeforePlay", "VisitorScoreBeforePlay"
3. Compress categories: StadiumType, Turf, GameWeather
"""

import numpy as np
import pandas as pd


class DataCleaner:

    def __init__(self, df):
        # Categorical Columns: update here to include stadium type
        self.categoricals = ["PossessionTeam", "FieldPosition", "HomeTeamAbbr", "VisitorTeamAbbr",
                             "OffenseFormation", "OffensePersonnel", "DefensePersonnel", "Down",
                             "Quarter", "Turf", "GameWeather"]
        self.categories = {col: [val for val in df[col].unique() if pd.notna(val)]
                           for col in self.categoricals}
        # fixing abbreviations
        self.map_Teams = {"ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU"}

        # don't miss a team and all categories for teams must match
        self.col_teams = ["PossessionTeam", "FieldPosition", "HomeTeamAbbr", "VisitorTeamAbbr"]
        for col in self.col_teams:
            for team in self.map_Teams.keys():
                self.categories[col][self.categories[col] == team] = self.map_Teams[team]
        teams = set().union(self.categories["PossessionTeam"], self.categories["FieldPosition"],
                            self.categories["HomeTeamAbbr"], self.categories["VisitorTeamAbbr"])
        for col in self.col_teams:
            self.categories[col] = teams

        # Need better defaults: NE may not even be playing...
        # Should see which features are relevant before worrying about this
        self.categorical_imputeVal = {
            'PossessionTeam': 'NE', 'FieldPosition': 'BUF',
            'HomeTeamAbbr': 'SF', 'VisitorTeamAbbr': 'LA',
            'OffenseFormation': 'SINGLEBACK',
            'OffensePersonnel': '1 RB, 1 TE, 3 WR',
            'DefensePersonnel': '4 DL, 2 LB, 5 DB',
            'Down': 1, 'Quarter': 1,
            'Turf': 'grass', 'GameWeather': 'overcast'
        }

        # Columns to Delete:
        self.dropColumns = ["GameClock", "YardLine", "DisplayName", "JerseyNumber", "Season",
                            "PlayerBirthDate", "PlayerCollegeName", "Stadium",
                            "Location", "WindSpeed", "WindDirection",
                            "HomeScoreBeforePlay", "VisitorScoreBeforePlay", "Humidity",
                            "Temperature"]

        # StadiumType --> expect irrelevant
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
        self.map_StadiumType = {nameList[i][j]: newNames[i] for i in range(len(newNames))
                                for j in range(len(nameList[i]))}

        # DefendersInTheBox
        self.defenders_imputeVal = df.DefendersInTheBox.mean()

        # Orientation
        self.orientation_imputeVal = df.Orientation.mean()

        # Dir
        self.dir_imputeVal = df.Dir.mean()

        # Turf --> expect irrelevant
        turfs = ['Field Turf', 'A-Turf Titan', 'Grass', 'UBU Sports Speed S5-M',
                 'Artificial', 'DD GrassMaster', 'Natural Grass',
                 'UBU Speed Series-S5-M', 'FieldTurf', 'FieldTurf 360',
                 'Natural grass', 'grass', 'Natural', 'Artifical', 'FieldTurf360',
                 'Naturall Grass', 'Field turf', 'SISGrass',
                 'Twenty-Four/Seven Turf', 'natural grass']
        grass = ["Grass", "DD GrassMaster", "Natural Grass", "Natural grass", "grass",
                 "Natural", "Naturall Grass", "natural grass"]
        artificial = [g for g in turfs if g not in grass]
        newNames = ["grass", "artificial"]
        nameList = [grass, artificial]
        self.map_Turf = {nameList[i][j]: newNames[i] for i in range(len(newNames))
                         for j in range(len(nameList[i]))}
        self.categories["Turf"] = newNames

        # GameWeather --> expect only snow is relevant, maybe heavy rain
        rain = ['Rainy', 'Rain Chance 40%', 'Showers',
                'Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.',
                'Scattered Showers', 'Cloudy, Rain', 'Rain shower', 'Light Rain', 'Rain']
        overcast = ['Party Cloudy', 'Cloudy, chance of rain',
                    'Coudy', 'Cloudy, 50% change of rain', 'Rain likely, temps in low 40s.',
                    'Cloudy and cold', 'Cloudy, fog started developing in 2nd quarter',
                    'Partly Clouidy', '30% Chance of Rain', 'Mostly Coudy', 'Cloudy and Cool',
                    'cloudy', 'Partly cloudy', 'Overcast', 'Hazy', 'Mostly cloudy', 'Mostly Cloudy',
                    'Partly Cloudy', 'Cloudy']
        clear = ['Partly clear', 'Sunny and clear', 'Sun & clouds', 'Clear and Sunny',
                 'Sunny and cold', 'Sunny Skies', 'Clear and Cool', 'Clear and sunny',
                 'Sunny, highs to upper 80s', 'Mostly Sunny Skies', 'Cold',
                 'Clear and warm', 'Sunny and warm', 'Clear and cold', 'Mostly sunny',
                 'T: 51; H: 55; W: NW 10 mph', 'Clear Skies', 'Clear skies', 'Partly sunny',
                 'Fair', 'Partly Sunny', 'Mostly Sunny', 'Clear', 'Sunny', "Sunny, Windy"]
        snow = ['Heavy lake effect snow', 'Snow', 'Cloudy, light snow accumulating 1-3""']
        indoor = ['N/A Indoor', 'Indoors', 'Indoor', 'N/A (Indoors)', 'Controlled Climate', "N/A"]
        newNames = ["rain", "overcast", "clear", "snow", "indoor"]
        nameList = [rain, overcast, clear, snow, indoor]
        self.map_GameWeather = {nameList[i][j]: newNames[i] for i in range(len(newNames))
                                for j in range(len(nameList[i]))}
        self.categories["GameWeather"] = newNames

    def create_categoricals(self, df):
        df.replace(self.map_Teams)

        for col in self.categoricals:
            # fill missing or new categories with most frequent
            df.loc[~df[col].isin(self.categories[col]), col] = self.categorical_imputeVal[col]
            df[col] = pd.Categorical(df[col], categories=self.categories[col]).codes

    def clean_data(self, df):
        # compute player height
        temp = df.PlayerHeight.str.split("-", expand=True)
        df.PlayerHeight = pd.to_numeric(temp[0]) * 12 + pd.to_numeric(temp[1])

        # away or home
        df["Team"] = df["Team"].apply(lambda x: 0 if x == "away" else 1)

        # missing defenders values imputed
        df.DefendersInTheBox.fillna(self.defenders_imputeVal, inplace=True)

        # missing orientation imputed
        df.Orientation.fillna(self.orientation_imputeVal, inplace=True)

        # missing dir values imputed
        df.Dir.fillna(self.dir_imputeVal, inplace=True)

        # missing field position occurs when on 50 yard line
        df.FieldPosition = np.where(df.YardLine == 50, df.PossessionTeam, df.FieldPosition)

        # clean stadium types; currently only consider outdoor
        df.fillna({"StadiumType": "Other"}, inplace=True)
        df["StadiumType"] = df["StadiumType"].map(self.map_StadiumType, na_action='ignore')
        df["StadiumType"] = df["StadiumType"].apply(lambda x: 1 if x == "outdoor" else 0)

        # Turf
        df.fillna({"Turf": "Other"}, inplace=True)
        df["Turf"] = df["Turf"].map(self.map_Turf, na_action='ignore')

        # GameWeather
        df.fillna({"GameWeather": "Other"}, inplace=True)
        df["GameWeather"] = df["GameWeather"].map(self.map_GameWeather, na_action='ignore')

        # update categoricals in-place, done after collapsing categories
        self.create_categoricals(df)

        # generating features that don't depend on play grouping
        df["DistanceToGoal"] =\
            df.YardLine + (100 - 2 * df.YardLine) * (df.PossessionTeam == df.FieldPosition)

        los_condition = ((df.PlayDirection == "right") & (df.PossessionTeam == df.FieldPosition)) \
            | ((df.PlayDirection == "left") & (df.PossessionTeam != df.FieldPosition))
        los_side = np.where(los_condition, "left", "right")
        df["LineOfScrimmage"] = \
            np.where(los_side == "left", df.YardLine + 10, 110 - df.YardLine)

        # sort and drop irrelevant columns
        df.sort_values(by=["GameId", "PlayId"]).reset_index()
        df.set_index("PlayId", inplace=True)
        df.drop(self.dropColumns, inplace=True, axis=1)
        return df


if __name__ == "__main__":
    data = pd.read_csv('../input/train.csv', low_memory=False)
    # df = data.copy(deep=True)
    cleaner = DataCleaner(data)
    dfClean = cleaner.clean_data(data)
    for c in dfClean.columns:
        print(dfClean[c].sample(10))
