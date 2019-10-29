
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

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
            xi, yi = self.play_to_array(dfs)
            x.append(xi)
            y.append(yi)

        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        return x, y

    def play_to_array(self, df):
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


