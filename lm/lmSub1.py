# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import norm

# from kaggle.competitions import nflrush
from input.kaggle.competitions import nflrush
env = nflrush.make_env()

################################################################################
# Cleaner
################################################################################


class DataCleaner:

    def __init__(self, df):
        # Categorical Columns: update here to include stadium type
        self.categoricals = ["PossessionTeam", "FieldPosition", "HomeTeamAbbr", "VisitorTeamAbbr",
                             "OffenseFormation", "OffensePersonnel", "DefensePersonnel", "Down",
                             "Quarter", "Turf", "GameWeather"]
        self.categories = {col: [val for val in df[col].unique() if pd.notna(val)]
                           for col in self.categoricals}
        # fixing abbreviations
        # self.map_Teams = {"ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU"}
        self.map_Teams = {"ARI": "ARZ", "BAL": "BLT", "CLE": "CLV", "HOU": "HST"}

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
                            "Temperature", "Team"]

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
        for col in self.categoricals:
            # fill missing or new categories with most frequent
            df.loc[~df[col].isin(self.categories[col]), col] = self.categorical_imputeVal[col]
            df[col] = pd.Categorical(df[col], categories=self.categories[col]).codes

    @staticmethod
    def rescale_location(df):
        # X will be in (-10,110); LOS will be in (0,100); DistToGoal = 100 - LOS (so dropped)
        df["LineOfScrimmage"] = np.where(df.FieldPosition == df.PossessionTeam,
                                         df.YardLine, 50 + (50 - df.YardLine))
        df["X"] = np.where(df.PlayDirection == "left", 120 - df.X, df.X) - 10  # replace
        df["Y"] = np.where(df.PlayDirection == "left", 160 / 3 - df.Y, df.Y)  # repace
        # df["DistanceToGoal"] =\
        #     df.YardLine + (100 - 2 * df.YardLine) * (df.PossessionTeam == df.FieldPosition)

    @staticmethod
    def rescale_dir_orient(df):
        """
        After rescaling, at 0 player moves left, 90 is straight, 180 to right, and 270 is backward.
        Dir is also converted to to pi*rad from so that it's easier to use in later files.
        Elsewhere: to compute using sin/cos, use (90-Dir)*np.pi/180
        """
        dir_temp = np.where((df.PlayDirection == "left") & (df.Dir < 90), df.Dir + 360, df.Dir)
        dir_temp = np.where((df.PlayDirection == "right") & (df.Dir > 270), df.Dir - 360, dir_temp)
        df["Dir"] = np.where(df.PlayDirection == "left", dir_temp - 180, dir_temp)
        or_temp = np.where((df.PlayDirection == "left") & (df.Orientation < 90),
                           df.Orientation + 360, df.Orientation)
        or_temp = np.where((df.PlayDirection == "right") & (df.Orientation > 270),
                           df.Orientation - 360, or_temp)
        df["Orientation"] = np.where(df.PlayDirection == "left", or_temp - 180, or_temp)

    def clean_data(self, df):
        # compute player height
        temp = df.PlayerHeight.str.split("-", expand=True)
        df.PlayerHeight = pd.to_numeric(temp[0]) * 12 + pd.to_numeric(temp[1])

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
        df = df.replace({"VisitorTeamAbbr": self.map_Teams, "HomeTeamAbbr": self.map_Teams})
        self.create_categoricals(df)

        # rescale location; adds LineOfScrimmage
        self.rescale_location(df)

        # rescale direction and orientation
        self.rescale_dir_orient(df)

        # computing features which don't depend on play
        offense = np.where(df.PossessionTeam == df.HomeTeamAbbr, "home", "away")
        df["OnOffense"] = df.Team.values == offense

        # sort and drop irrelevant columns
        df.sort_values(by=["GameId", "PlayId"]).reset_index()
        df.set_index("PlayId", inplace=True)
        df.drop(self.dropColumns, inplace=True, axis=1)
        return df

################################################################################
# Feature Generator
################################################################################


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

    def specialYardIndicators(self, df):
        # First and ten/15/20
        pass

    def timeTillHandoff(self, df):
        pass

    @staticmethod
    def f_Team(dfP):
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

    def f_Rusher(self, dfP):
        # set up
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

        # output
        d = {"DistLOS": DistLOS, "DistDef": DistDef, "Acc": Acc,
             "SpeedX": SpeedX, "SpeedY": SpeedY, "Pos": Pos,
             "DistOffMean": DistOffMean, "DistDefMean": DistDefMean,
             "ADef": ADef, "SDef": SDef,
             "Gap": self.Gap(s, DistLOS, offense, defense)}
        return d

    @staticmethod
    def Gap(s, DistLOS, offense, defense, gapMult=1):
        # set up: compute gap location and size. Running toward edge if gap isn't entirely in field.
        Dir = s.Dir.values[0]
        ToEdge = 1
        if 0 <= Dir < 180:
            angle = min(Dir, 180 - Dir)
            deltaY = DistLOS / np.tan(angle * np.pi / 180.0)
            GapCenter = s.Y.values[0] + (-1)**(Dir > 90) * deltaY
            GapRadius = gapMult * DistLOS
            if GapCenter - GapRadius > 0 and GapCenter + GapRadius < 160 / 3:
                ToEdge = 0
        if 180 <= Dir <= 360 or ToEdge:
            side = "up" if 270 <= Dir or Dir < 90 else "down"
            GapCenter = (160 / 3 + s.Y.values[0]) / 2 if side == "up" else s.Y.values[0] / 2
            GapRadius = np.abs(GapCenter - s.Y.values[0])
            ToEdge = 1

        # compute statistics
        off_DistToGap = np.sqrt((offense.X - offense.LineOfScrimmage)**2 +
                                (offense.Y - GapCenter)**2)
        def_DistToGap = np.sqrt((defense.X - defense.LineOfScrimmage) ** 2 +
                                (defense.Y - GapCenter) ** 2)
        nOff = (off_DistToGap < GapRadius).sum()
        nDef = (def_DistToGap < GapRadius).sum()
        NPlayers = nOff + nDef
        AveSpace = NPlayers / GapRadius
        TeamRatio = (nOff + 1)/(nDef + 1)  # prevents /0
        defYLoc = defense[def_DistToGap < GapRadius].Y.to_list()
        defYLoc.sort()
        defYLoc.insert(0, GapCenter - GapRadius)
        defYLoc.append(GapCenter + GapRadius)
        OpenSize = np.diff(defYLoc).max()

        # output
        d = {"NPlayers": NPlayers, "AveSpace": AveSpace, "TeamRatio": TeamRatio,
             "OpenSize": OpenSize, "ToEdge": ToEdge, "Center": GapCenter, "Radius": GapRadius}
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


################################################################################
# Rescale Probabilities
################################################################################

def rescale_probabilities(probVec, lims=[-5, 25]):
    probVec = probVec.tolist()
    rangeY_true = np.arange(-99, 100).tolist()
    lower = rangeY_true.index(lims[0])
    upper = rangeY_true.index(lims[1])
    probNew = list()
    for index in range(len(probVec)):
        if index <= lower:
            probNew.append(0)
        elif index > upper:
            probNew.append(1)
        else:
            probNew.append(probVec[index]/probVec[upper-1])
    return probNew
rescale_probabilities(probs)
################################################################################
# Reading, Preprocessing, Fit Model
################################################################################


print('- read data')
# df_tr = pd.read_csv('./input/train.csv', low_memory=False)
df_tr = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)

print('- clean train data')
cleaner = DataCleaner(df_tr)
df_tr = cleaner.clean_data(df_tr)

print('- get features')
featgenerator = FeatureGenerator()
xtr, ytr, playid_tr = featgenerator.make_features(df_tr, test=False)
# xtr = pd.read_csv('./input/features_py.csv', low_memory=False).set_index("PlayId")
# ytr = pd.read_csv('./input/response_py.csv', low_memory=False).Yards


# transform y
ytr = np.log(np.clip(ytr - min(ytr) + 1, 0, None))

# transform x
toRemove = np.append(np.arange(11), np.arange(12, 15))
toRemove = np.append(toRemove, np.arange(19, 27))
toRemove = np.append(toRemove, 31)
xtr = xtr.drop(xtr.columns[toRemove], axis=1)

categoricalCl = ["Down", "OffenseFormation", "Turf", "GameWeather"]
categoricalFea = ["Rusher_Pos"]
categoriesFea = {col: [val for val in xtr[col].unique() if pd.notna(val)]
                 for col in categoricalFea}

for col in categoricalCl:
    xtr[col] = pd.Categorical(xtr[col], categories=cleaner.categories[col])
for col in categoricalFea:
    xtr[col] = pd.Categorical(xtr[col], categories=categoriesFea[col])
xtr = pd.get_dummies(xtr, prefix_sep="_", drop_first=True)

# get polynomials and fit model
polyFeat = PolynomialFeatures(degree=2)
xtr2 = polyFeat.fit_transform(xtr)
lm = LinearRegression()
lm.fit(xtr2, ytr)
rangeY = np.log(np.clip(np.arange(-99, 100) + 15, 0, None))

# for (df_te, sample_prediction_df) in env.iter_test():
#     pass
# lm.predict(xtr2)
for i, (df_te, sample_prediction_df) in enumerate(env.iter_test()):
    if i % 100 == 0:
        print('  - Iter %d' % (i + 1))

    df_te = cleaner.clean_data(df_te)
    xte, playid_te = featgenerator.make_features(df_te, test=True)
    # df = df.reindex(sorted(df.columns), axis=1)
    # process for lm
    xte = xte.drop(xte.columns[toRemove], axis=1)
    for col in categoricalCl:
        xte[col] = pd.Categorical(xte[col], categories=cleaner.categories[col])
    for col in categoricalFea:
        xte[col] = pd.Categorical(xte[col], categories=categoriesFea[col])
    xte = pd.get_dummies(xte, prefix_sep="_", drop_first=True)

    # get predictions and cdf
    xte2 = polyFeat.fit_transform(xte)
    pred = lm.predict(xte2).__float__()
    probs = norm.cdf(rangeY, pred, .2).reshape(1, -1)
    df_pred = pd.DataFrame(data=probs, columns=sample_prediction_df.columns)
    env.predict(df_pred)

env.write_submission_file()
print('Done!!!')  # This Python 3 environment comes with many helpful analytics libraries installed
