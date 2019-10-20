
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, RandomSampler


#### danijels/korys network

class nn1(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, batch_size, n_unique):
        super(nn1, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.embeds = [nn.Embedding(n, 2) for n in n_unique]
        self.relu = nn.ReLU()
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.parameters(), lr=.001)

    def forward(self, input):
        x_num, x_cat = input
        x_cat = [embedding(x_cat[:, i]) for i, embedding in enumerate(self.embeds)]
        x_cat = torch.cat(x_cat, 1)
        x = torch.cat((x_num, x_cat), 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        out = []
        out.append(self.relu(x[:, [0]]))
        for i in range(1, 199): out.append(out[i-1] + self.relu(x[:,[i]]-out[i-1]))
        out = torch.cat(out, 1)
        return out

    def compute_loss(self, X, y):
        output = self.forward(X)
        loss = torch.mean(torch.pow(output-y, 2))
        return loss

    def train_epochs(self, nepochs, X, y):
        for _ in range(nepochs):
            batches = BatchSampler(RandomSampler(range(len(X[0]))), batch_size=self.batch_size, drop_last=False)
            for batch in batches:
                self.train()
                X_batch = [X[0][batch,:], X[1][batch,:]]
                y_batch = y[batch,:]
                loss = self.compute_loss(X_batch, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def eval_model(self, X, y):
        self.eval()
        loss = self.compute_loss(X, y)
        return loss.item()

    def predict(self, X):
        self.eval()
        output = self.forward(X)
        return output


class nn2(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, batch_size, n_unique):
        super(nn2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.embeds = [nn.Embedding(n, 2) for n in n_unique]
        self.relu = nn.ReLU()
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.parameters(), lr=.001)

    def forward(self, input):
        x_num, x_cat = input
        x_cat = [embedding(x_cat[:, i]) for i, embedding in enumerate(self.embeds)]
        x_cat = torch.cat(x_cat, 1)
        x = torch.cat((x_num, x_cat), 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        out0 = x[:, [0]]
        out1 = self.relu(x[:, [1]]) + .025
        out = torch.cat((out0, out1), 1)
        return out

    def compute_loss(self, X, y):
        output = self.forward(X)
        l1 = torch.log(output[:, 1])
        l2 = (y - output[:, 0]).pow(2) / output[:, 1]
        loss = torch.mean(l1 + l2)
        return loss

    def train_epochs(self, nepochs, X, y):
        for _ in range(nepochs):
            batches = BatchSampler(RandomSampler(range(len(X[0]))), batch_size=self.batch_size, drop_last=False)
            for batch in batches:
                self.train()
                X_batch = [X[0][batch,:], X[1][batch,:]]
                y_batch = y[batch,:]
                loss = self.compute_loss(X_batch, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def eval_model(self, X, y):
        self.eval()
        loss = self.compute_loss(X, y)
        return loss.item()

    def predict(self, X):
        self.eval()
        output = self.forward(X)
        return output





class Preprocessor:

    def __init__(self, x, y, feat_num, feat_cat):
        self.scaler = StandardScaler()
        self.feat_num = feat_num
        self.feat_cat = feat_cat
        self.scaler.fit(x[self.feat_num])

    def transform_data(self, x, y, is_train):
        x[self.feat_num] = self.scaler.transform(x[self.feat_num])
        x_num = torch.from_numpy(x[self.feat_num].values).float()
        x_cat = torch.from_numpy(x[self.feat_cat].values).long()

        x = [x_num, x_cat]
        y = torch.from_numpy(pd.get_dummies(pd.Categorical(y.iloc[:,0] + 99, range(199))).cumsum(axis=1).values.astype(float)).float() if is_train else []

        return x, y


class Preprocessor_nn2:

    def __init__(self, x, y, feat_num, feat_cat):
        self.scaler = StandardScaler()
        self.feat_num = feat_num
        self.feat_cat = feat_cat
        self.scaler.fit(x[self.feat_num])

    def transform_data(self, x, y, is_train):
        x[self.feat_num] = self.scaler.transform(x[self.feat_num])
        x_num = torch.from_numpy(x[self.feat_num].values).float()
        x_cat = torch.from_numpy(x[self.feat_cat].values).long()

        x = [x_num, x_cat]
        y = torch.from_numpy(np.log(y.clip(-14, 99) + 15).values.astype(float)).float() if is_train else []

        return x, y


def eval_model(y_mat, predicted):
    yrange = np.log(np.arange(-99, 100).clip(-14, 99) + 15)
    yrange = np.tile(yrange, (len(y_mat),1))
    mean = np.tile(predicted[:,[0]], (1,199))
    sd = np.tile(predicted[:,[1]], (1,199))
    pred = norm.cdf(yrange, mean, sd)
    loss = np.mean((y_mat-pred)**2)
    return loss

