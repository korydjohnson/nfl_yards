
import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter as gfilt

import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler


#### danijels/korys network
class convnet(nn.Module):

    def __init__(self, out_channels_1, out_channels_2,  batch_size, learning_rate):
        super(convnet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=out_channels_1, kernel_size=(3, 3), stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=out_channels_2, kernel_size=(3, 3), stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.fc = nn.Linear(105, 199)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = x.reshape((-1, 105))
        x = self.fc(x)
        out = self.sigmoid(x)
        return out

    def compute_loss(self, X, y):
        output = self.forward(X)
        loss = torch.mean(torch.pow(output-y, 2))
        return loss

    def train_epochs(self, nepochs, gen):
        for _ in range(nepochs):
            len(gen.playids)
            batches = BatchSampler(RandomSampler(range(len(gen.playids))), batch_size=self.batch_size, drop_last=False)
            for batch in batches:
                self.train()
                xbatch, ybatch = gen.get_features(batch)
                loss = self.compute_loss(xbatch, ybatch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def eval_model(self, gen, batch_size):
        n = len(gen.playids)
        batches = BatchSampler(SequentialSampler(range(n)), batch_size=batch_size, drop_last=False)
        l = 0
        for batch in batches:
            xbatch, ybatch = gen.get_features(batch)
            self.eval()
            loss = self.compute_loss(xbatch, ybatch)
            l += len(batch)*loss
        l /= n
        return l

    def predict(self, X):
        self.eval()
        output = self.forward(X)
        return output


class DataGenerator:

    def __init__(self, df, playids):
        self.df = df
        self.playids = playids

    def get_features(self, index):
        ids = self.playids[index]
        x = []
        y = []
        for id in ids:
            dfs = self.df[self.df['PlayId'] == id].reset_index(drop=True)
            # xi, yi = self.play_to_array(dfs)
            xi, yi = self.play_to_heatmap(dfs)
            x.append(xi)
            y.append(yi)

        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        return x, y

    @staticmethod
    def play_to_array(df):
        rusher = df[df['NflId'] == df['NflIdRusher']]
        offense = df[df.OnOffense]
        defense = df[~df.OnOffense]
        x = np.zeros((1, 3, 120, 60))

        xpos = rusher['X'].round().astype(int).values
        ypos = rusher['Y'].round().astype(int).values
        x[0, 0, xpos, ypos] = 1

        xpos = offense['X'].round().astype(int).values
        ypos = offense['Y'].round().astype(int).values
        x[0, 1, xpos, ypos] = 1

        xpos = defense['X'].round().astype(int).values
        ypos = defense['Y'].round().astype(int).values
        x[0, 2, xpos, ypos] = 1

        tmp = np.zeros((1, 199))
        tmp[0, rusher['Yards'] + 99] = 1
        y = tmp.cumsum(axis=1)

        return x, y

    def player_vec(self, player, times, LOS, scale=True):
        dirRad = (90 - player.Dir) * np.pi % 180
        speedVec = player.S + player.A*times

        xVec = player.X + np.cos(dirRad)*(player.S*times + player.A*times**2/2)
        xVec = np.clip(xVec.round(), -10, 110).astype(int)
        xVec = np.clip(xVec - LOS + 21, 0, 41)  # crop to 41 yard field

        yVec = player.Y + np.sin(dirRad)*(player.S*times + player.A*times**2/2)
        yVec = np.clip(yVec.round(), 0, 53).astype(int)

        wVec = (1 - speedVec/sum(speedVec))/2 if sum(speedVec) > 0 else np.repeat(1, len(times))
        wVec = wVec/max(wVec) if scale else wVec

        return xVec, yVec, wVec

    def play_to_heatmap(self, dfP, filt=False, test=False, s=.5, w=3, nPoints=3):
        times = np.linspace(0, 1, nPoints+2)
        rusher = dfP[dfP['NflId'] == dfP['NflIdRusher']]
        LOS = rusher.LineOfScrimmage.values[0]
        offense = dfP[dfP.OnOffense]
        defense = dfP[~dfP.OnOffense]
        x = np.zeros((1, 3, 42, 54))

        # fill player location vectors
        xpos, ypos, w = self.player_vec(rusher.squeeze(), times, LOS)
        x[0, 0, xpos, ypos] = w

        for player in offense.itertuples(index=False):
            xpos, ypos, w = self.player_vec(player, times, LOS)
            x[0, 1, xpos, ypos] = w

        for player in defense.itertuples(index=False):
            xpos, ypos, w = self.player_vec(player, times, LOS)
            x[0, 2, xpos, ypos] = w

        if filt:  # filter
            t = (((w - 1) / 2) - 0.5) / s
            for dim in range(3):
                x[0, dim, :, :] = gfilt(x[0, dim, :, :], sigma=s, truncate=t)

        if test:
            return x
        else:
            tmp = np.zeros((1, 199))
            tmp[0, rusher.Yards + 99] = 1
            y = tmp.cumsum(axis=1)
            return x, y


if __name__ == "__main__":
    df_tr = pd.read_csv('./input/trainClean_py.csv', low_memory=False)
    playids_tr = df_tr.PlayId
    tr = DataGenerator(df_tr, playids_tr)
    dfSub = df_tr[df_tr.PlayId == 20181230154157]
    self = tr
    dfP = dfSub
    # tr.play_to_heatmap(dfSub, test=False, s=.5, nPoints=3, filt=False)
    # x, y = tr.get_features(np.arange(1, len(playids_tr)))

    import matplotlib.pyplot as plt
    x, y = tr.play_to_heatmap(dfSub, test=False, s=.5, nPoints=3, filt=False)
    x = x[0, :, :, :]
    a = []
    for i in range(3):
        a.append(x[i, :, :])
    a = np.stack(a, axis=2)

    plt.imshow(a)
    plt.show()
