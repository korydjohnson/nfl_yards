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