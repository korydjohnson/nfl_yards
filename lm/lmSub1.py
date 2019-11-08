"""
Polynomial Model
"""

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.isotonic import IsotonicRegression
from scipy.stats import norm
from scipy.ndimage.filters import gaussian_filter as gfilt
import gc


from input.kaggle.competitions import nflrush
# from kaggle.competitions import nflrush
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
                            "Temperature", "Team", "TimeSnap", "TimeHandoff"]

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
        df["TimeToHandoff"] = (pd.to_datetime(df.TimeHandoff) - pd.to_datetime(df.TimeSnap)) \
            / np.timedelta64(1, 's')

        # truncate features
        df.S = np.clip(df.S, .5, 10)

        # sort and drop irrelevant columns
        df.sort_values(by=["GameId", "PlayId"]).reset_index()
        df.set_index("PlayId", inplace=True)
        df.drop(self.dropColumns, inplace=True, axis=1)
        return df


################################################################################
# Feature Generator
################################################################################

class ImageGenerator:

    def __init__(self, filt=False, s=.5, width=3, nPoints=1, times=None):
        self.filt = filt
        self.s = s
        self.width = width
        self.nPoints = nPoints
        if times is None:
            self. times = np.linspace(0, 1, self.nPoints+2)
        else:
            self.times = np.array(times)

    def get_images(self, df):
        images = []
        gc.disable()
        for play in df.index.unique():
            dfP = df.filter(like=str(play), axis=0)
            image = self.play_to_heatmap(dfP)
            images.append(image)
        gc.enable()

        return np.concatenate(images, axis=0)

    def player_vec(self, player, LOS):
        dirRad = (90 - player.Dir) * np.pi % 180

        xVec = player.X + np.cos(dirRad)*(player.S*self.times + player.A*self.times**2/2)
        xVec = np.clip(xVec.round(), -10, 110)
        xVec = np.clip(xVec - LOS + 21, 0, 41).astype(int)  # crop to 41 yard field

        yVec = player.Y + np.sin(dirRad)*(player.S*self.times + player.A*self.times**2/2)
        yVec = np.clip(yVec.round(), 0, 53).astype(int)

        wVec = abs(1 / (self.times + 1))
        wVec = wVec/max(wVec)

        return xVec, yVec, wVec

    def play_to_heatmap(self, dfP):
        rusher = dfP[dfP['NflId'] == dfP['NflIdRusher']].squeeze()
        LOS = rusher.LineOfScrimmage
        offense = dfP[dfP.OnOffense]
        defense = dfP[~dfP.OnOffense]
        image = np.zeros((1, 3, 42, 54))

        # fill player location vectors
        xpos, ypos, w = self.player_vec(rusher, LOS)
        image[0, 0, xpos, ypos] = w

        for player in offense.itertuples(index=False):
            xpos, ypos, w = self.player_vec(player, LOS)
            image[0, 1, xpos, ypos] = w

        for player in defense.itertuples(index=False):
            xpos, ypos, w = self.player_vec(player, LOS)
            image[0, 2, xpos, ypos] = w

        if self.filt:  # filter
            t = (((self.width - 1) / 2) - 0.5) / self.s
            for dim in range(3):
                image[0, dim, :, :] = gfilt(image[0, dim, :, :], sigma=self.s, truncate=t)

        return image


class FeatureGenerator:
    def __init__(self, images=True, features=None, filt=False, s=.5, width=3,
                 nPoints=3, times=None):
        self.images = images
        # creating features, method names
        self.features = features
        self.response = "Yards"
        # repeated features; PlayId excluded as it'rusher the index; Yards for response
        self.repeated_features = ['Quarter', 'PossessionTeam', 'Down',
                                  'Distance', 'OffenseFormation', 'OffensePersonnel',
                                  'DefendersInTheBox', 'DefensePersonnel', 'HomeTeamAbbr',
                                  'VisitorTeamAbbr', 'Week', 'StadiumType', 'Turf', 'GameWeather',
                                  'LineOfScrimmage']
        self.dropColumns = ["GameId", "X", "Y", "S", "A", "Dis", "Orientation",
                            "Dir", "NflId", "FieldPosition", "PlayerHeight", "PlayerWeight",
                            "Position", "HomeTeamAbbr", "VisitorTeamAbbr", "Week", "StadiumType",
                            "OnOffense", "NflIdRusher", "PossessionTeam", "Quarter",
                            "PlayDirection"]
        self.categoricals = ["Down", "OffenseFormation", "OffensePersonnel", "DefensePersonnel",
                             "Rusher_Pos", "Turf", "GameWeather", "Rusher_Gap_ToEdge"]
        self.feature_categorical = ["Rusher_Pos"]  # constructed feature; cleaned at end
        self.images = images
        if images:
            self.imageGen = ImageGenerator(filt, s, width, nPoints, times)

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
        rusher = dfP[dfP['NflId'] == dfP['NflIdRusher']].squeeze()
        rush_dir_rad = (90 - rusher.Dir) * np.pi / 180.0
        offense = dfP[dfP.OnOffense].copy()
        defense = dfP[~dfP.OnOffense].copy()
        dist_off = np.sqrt((offense.X - rusher.X)**2 + (offense.Y - rusher.Y)**2).values
        dist_def = np.sqrt((defense.X - rusher.X)**2 + (defense.Y - rusher.Y)**2).values

        # descriptive statistics
        DistDef = dist_def.min()
        closest_opponent = defense.iloc[np.argmin(dist_def)]
        ADef = closest_opponent.A-rusher.A
        SDef = closest_opponent.S-rusher.S
        Acc = rusher.A
        SpeedX = np.abs(rusher.S * np.cos(rush_dir_rad))
        SpeedY = np.abs(rusher.S * np.sin(rush_dir_rad))
        Pos = rusher.Position

        # interpretive/computed statistics
        DistOffMean = dist_off.mean() * 11 / 10  # rescale for rusher 0
        DistDefMean = dist_def.mean()
        DistLOS = rusher.LineOfScrimmage - rusher.X

        # output
        d = {"DistLOS": DistLOS, "DistDef": DistDef, "Acc": Acc,
             "SpeedX": SpeedX, "SpeedY": SpeedY, "Pos": Pos,
             "DistOffMean": DistOffMean, "DistDefMean": DistDefMean,
             "ADef": ADef, "SDef": SDef,
             "Gap": self.Gap(rusher, DistLOS, offense, defense)}
        return d

    @staticmethod  # gapMult=1 sets gap radius as 1 second; or gapRadius*rusherSpeed = distLOS
    def Gap(rusher, DistLOS, offense, defense, gapMult=1):
        # set up: compute gap location and size. Running toward edge if gap isn't entirely in field.
        Dir = rusher.Dir
        ToEdge = 1
        if 0 <= Dir < 180:
            angle = min(Dir, 180 - Dir)
            deltaY = DistLOS / np.tan(angle * np.pi / 180.0)
            GapCenter = rusher.Y + (-1)**(Dir > 90) * deltaY
            if 0 < GapCenter < 160 / 3:  # prev checked if entire ball in field
                ToEdge = 0
        if 180 <= Dir <= 360 or ToEdge:
            side = "up" if 270 <= Dir or Dir < 90 else "down"
            GapCenter = (160 / 3 + rusher.Y) / 2 if side == "up" else rusher.Y / 2
            # GapRadius = (np.abs(GapCenter - rusher.Y.values[0])) / rusher.S.values[0]
        DistDirLOS = np.sqrt((rusher.X - rusher.LineOfScrimmage) ** 2 + (rusher.Y - GapCenter) ** 2)
        GapRadius = (gapMult * DistDirLOS) / rusher.S

        # compute statistics; who *will be* in gap/ball at LOS. Either Distance or Time based.
        offense["X_end"] = offense.S * np.cos((90 - offense.Dir) * np.pi % 180) + offense.X
        offense["Y_end"] = offense.S * np.sin((90 - offense.Dir) * np.pi % 180) + offense.Y
        defense["X_end"] = defense.S * np.cos((90 - defense.Dir) * np.pi % 180) + defense.X
        defense["Y_end"] = defense.S * np.sin((90 - defense.Dir) * np.pi % 180) + defense.Y
        off_DistToGap = np.sqrt((offense.X_end - offense.LineOfScrimmage)**2 +
                                (offense.Y_end - GapCenter)**2)
        def_DistToGap = np.sqrt((defense.X_end - defense.LineOfScrimmage) ** 2 +
                                (defense.Y_end - GapCenter) ** 2)
        nOffT = (off_DistToGap / offense.S < GapRadius).sum()
        nDefT = (def_DistToGap / defense.S < GapRadius).sum()
        NPlayersT = nOffT + nDefT
        AveSpaceT = NPlayersT / GapRadius
        TeamRatioT = (nOffT + 1)/(nDefT + 1)  # prevents /0
        OpenSizeT = min(def_DistToGap / defense.S)  # minimum time to center of gap
        nOffD = (off_DistToGap < gapMult * DistDirLOS).sum()
        nDefD = (def_DistToGap < gapMult * DistDirLOS).sum()
        NPlayersD = nOffD + nDefD
        AveSpaceD = NPlayersD / (gapMult * DistDirLOS)
        TeamRatioD = (nOffD + 1) / (nDefD + 1)  # prevents /0
        OpenSizeD = min(def_DistToGap)  # minimum time to center of gap

        # output
        d = {"NPlayersT": NPlayersT, "AveSpaceT": AveSpaceT, "TeamRatioT": TeamRatioT,
             "OpenSizeT": OpenSizeT, "NPlayersD": NPlayersD, "AveSpaceD": AveSpaceD,
             "TeamRatioD": TeamRatioD, "OpenSizeD": OpenSizeD, "ToEdge": ToEdge,
             "Center": GapCenter, "Radius": GapRadius, "DistDirLOS": DistDirLOS}
        return d

    def set_standards(self, covariates):
        self.numeric = [col for col in covariates.columns if col not in self.categoricals]
        self.means = covariates[self.numeric].mean()
        self.sds = np.sqrt(covariates[self.numeric].var())
        self.feature_categories = {col: [v for v in covariates[col].unique() if pd.notna(v)]
                                   for col in self.feature_categorical}

    def standardize(self, covariates):
        covariates[self.numeric] = (covariates[self.numeric] - self.means) / self.sds
        for col in self.feature_categorical:
            covariates[col] = pd.Categorical(covariates[col],
                                             categories=self.feature_categories[col])

    def new_features(self, dfP, methods):
        out = {}
        for method in methods:
            out[method[2:]] = getattr(self, method)(dfP)  # always store features as dict
        out = pd.io.json.json_normalize(out, sep='_')
        return out.iloc[0]

    def make_features(self, df):
        if self.features is None:
            methods = [method for method in dir(self)
                       if callable(getattr(self, method)) if method.startswith('f_')]
        else:
            methods = ["f_" + feature for feature in self.features]

        out = df.drop_duplicates(self.repeated_features)
        extractedFeatures = df.groupby('PlayId').apply(self.new_features, methods)
        out = out.join(extractedFeatures)
        if self.images:
            extractedImages = self.imageGen.get_images(df)

        # return based on training or test data
        covariates = out.drop(columns=self.dropColumns)
        if self.response in out.columns:  # training set
            covariates = covariates.drop(columns=self.response)
            self.set_standards(covariates)
            self.standardize(covariates)
            if self.images:
                return extractedImages, covariates, out[self.response], out.index.values
            else:
                return covariates, out[self.response], out.index.values
        else:
            self.standardize(covariates)
            if self.images:
                return extractedImages, covariates, out.index.values
            else:
                return covariates, out.index.values


################################################################################
# Rescale Probabilities
################################################################################

def rescale_probabilities(probsVec, lims=[-5, 25]):
    rangeY_true = np.arange(-99, 100)
    lower = np.where(rangeY_true == lims[0])[0].__int__()
    upper = np.where(rangeY_true == lims[1])[0].__int__()
    probsSel = probsVec[lower:upper]
    probsSel = probsSel/max(probsSel)
    probsN = np.concatenate((np.repeat(0, lower, axis=0), probsSel,
                             np.repeat(1, 199 - upper, axis=0)), axis=0)
    return probsN

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
featgenerator = FeatureGenerator(images=False)
xtr, ytr, playid_tr = featgenerator.make_features(df_tr)
# xtr = pd.read_csv('./input/features_py.csv', low_memory=False).set_index("PlayId")
# ytr = pd.read_csv('./input/response_py.csv', low_memory=False).set_index("PlayId")

print('- lm prep')
# transform y
minY = -1 * min(ytr.Yards)
ytr = np.log(ytr + minY + 2)  # min val is log(2), use log(1) for truncated range


# transform x
categoricals = ["Down", "OffenseFormation", "OffensePersonnel", "DefensePersonnel",
                "Rusher_Pos", "Turf", "GameWeather", "Rusher_Gap_ToEdge"]
xtr = xtr.drop(xtr.columns[categoricals], axis=1)
# for col in categoricals:
#     xtr[col] = pd.Categorical(xtr[col], categories=cleaner.categories[col])
# xtr = pd.get_dummies(xtr, prefix_sep="_", drop_first=True)

print('- estimate model')
polyFeat = PolynomialFeatures(degree=2)
xtr2 = polyFeat.fit_transform(xtr)
lm = LinearRegression()
lm.fit(xtr2, ytr)
rangeY = np.log(np.clip(np.arange(-99, 100) + minY + 2, 1, None))

print('- recalibrate predictions')


def f(p):
    return np.mean(p1 < p)

recalibrateCdf(probs,)

p1 = norm.cdf((yval2 - output[:, 0]) / np.sqrt(output[:, 1]))



p2 = np.array([f(pi) for pi in p1])
ir = IsotonicRegression()
ir.fit(p1, p2)

# for (df_te, sample_prediction_df) in env.iter_test():
#     pass
# lm.predict(xtr2)
print('- make predictions on test set')
for i, (df_te, sample_prediction_df) in enumerate(env.iter_test()):
    if i % 100 == 0:
        print('  - Iter %d' % (i + 1))

    df_te = cleaner.clean_data(df_te)
    xte, playid_te = featgenerator.make_features(df_te, test=True)

    # process for lm
    xte = xte.drop(xte.columns[categoricals], axis=1)
    # for col in categoricals:
    #     xte[col] = pd.Categorical(xte[col], categories=cleaner.categories[col])
    # xte = pd.get_dummies(xte, prefix_sep="_", drop_first=True)

    # get predictions and cdf
    xte2 = polyFeat.fit_transform(xte)
    pred = lm.predict(xte2).__float__()
    probs = norm.cdf(rangeY, pred, .2)
    probsRe =
    df_pred = pd.DataFrame(data=probs.reshape(1, -1), columns=sample_prediction_df.columns)
    env.predict(df_pred)

env.write_submission_file()
print('Done!!!')  # This Python 3 environment comes with many helpful analytics libraries installed
