"""
Feature Generation Function(rusher)
1. Each function will define a new feature, a scaler per PLAY
2. Can assumed given clean data (see clean_data.py)
3. Final function calls all previous functions to generate new columns
4. In the end, will have one row per PLAY
5. Functions take entire play information and input is labelled dfP.

NOTE:
    All (finished) feature functions need to start with "f_" and will be called in
    ALPHABETICAL order. Since we are unnesting a dictionary, it is easy to have functions
    call other functions. These secondary functions cannot start with "f_".
NOTE:
    You can provide list of features to make_features, these are the feature names
    and need to match the functions. The "f_" will be appended in the function to
    make creating the feature list nicer/simpler.

END:
    Removed all "unnecessary" variables.
    Everythign is scaled except for self.categoricals.
    Parameters are set when creating feature constructor and running on training data. There is
    no longer a "test" argument, as it just checks whether or not "Yards" is in the column names.
    It also calls an image generator. You set the "times" argument to be the point or vector of
    times you'd like the image to cover. Default picks 3 points between 0 and 1,
    weight in image decreases as time increases.
"""

import pandas as pd
import numpy as np
from scipy.ndimage.filters import gaussian_filter as gfilt
import gc


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


if __name__ == "__main__":
    data = pd.read_csv('./input/trainClean_py.csv', low_memory=False).set_index("PlayId")
    ctor = FeatureGenerator()  # feature constructor
    dfSub = pd.concat([data.filter(like="20170907000118", axis=0),
                       data.filter(like="20181230154157", axis=0)])
    ctor.make_features(dfSub)
    dfSub = dfSub.drop("Yards", axis=1)
    ctor.make_features(dfSub)

    ims, features, playid = ctor.make_features(dfSub)
    features[ctor.numeric].mean()
    features[ctor.numeric].var()
    # image_ctor = ImageGenerator(times=[0])
    # images = image_ctor.play_to_heatmap(dfSub)
    import matplotlib.pyplot as plt
    im = ims[0, :, :, :]
    a = []
    for i in range(3):
        a.append(im[i, :, :])
    a = np.stack(a, axis=2)
    plt.imshow(a)
    plt.show()

    ims, x, y, PlayId = ctor.make_features(data)
    # x.head()
    # for c in x.columns:
    #     print(x[c].sample(10))
    x.to_csv("./input/features_py.csv")
    # y.to_csv("../input/response_py.csv")

    # make features for test data
    data = pd.read_csv('./input/testClean_py.csv', low_memory=False).set_index("PlayId")
    ctor = FeatureGenerator()  # feature constructor
    x, PlayId = ctor.make_features(data, test=True)
    x.head()
    x.to_csv("./input/featuresTest_py.csv")
