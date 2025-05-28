import torch
import __main__
from torch import nn
import numpy as np
import pandas as pd
import torch.nn.functional as F

torch.manual_seed(170798)

""" Simple network class with weights initialized to ~0 """
class SimpleNN(nn.Module):
    def __init__(self, inputs=3, layers=3, nodes=[10, 10]):
        super(SimpleNN, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(inputs, nodes[0])])

        for i in range(layers - 2):
            self.layers.append(nn.Linear(nodes[i], nodes[i+1]))

        self.layers.append(nn.Linear(nodes[-1], 1))  # Output layer

        self.sigmoid = nn.Sigmoid()
        self._initialize_weights()

    def _initialize_weights(self):
        """Custom weight initialization to make initial outputs close to 0."""
        for layer in self.layers[:-1]:  # Hidden layers
            nn.init.uniform_(layer.weight, -0.01, 0.01)  # Small random weights
            nn.init.constant_(layer.bias, 0)  # Bias set to zero

        # Output layer: Initialize bias to a large negative value
        nn.init.uniform_(self.layers[-1].weight, -0.01, 0.01)  # Small weights
        nn.init.constant_(self.layers[-1].bias, -15)  # Bias set to -15 to push output to 0

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.sigmoid(self.layers[-1](x))  # Sigmoid activation
        return x

""" Constructing the estimated MFPT as the loss function """
class lossMFPT(nn.Module):
    def __init__(self):
        super(lossMFPT, self).__init__()

    def forward(self, model, nTrajectories, X, **kwargs):
        finishedNotPassed = 0
        Ppassed = torch.zeros(nTrajectories)
        passTime = torch.zeros(nTrajectories)
        for i in range(len(X)):
            endProbability = model(X[i])[:,0] # The probability of resetting given that no resetting occurred previously
            survival = torch.cumprod(1 - endProbability, dim=0) / (1 - endProbability) # Survival probability of trajectory i through Eq. 4
            finishedNotPassed += (torch.Tensor(np.linspace(1, len(X[I]) + 1, len(X[i]))) * survival * endProbability).sum() # Adding the contribution of trajectory i in Eq. 7
            Ppassed[i] = survival[-1]
            passTime[i] = len(X[i])
        time = (finishedNotPassed + (passTime * Ppassed).sum()) / (Ppassed.sum()) # The MFPT through Eq. 3
        return time

""" Class for training the model """
class trainer():
    def __init__(self):
        self.loss = lossMFPT()

    def train_batch(self, nTrajectories,X, model, optimizer, lossFunc, kwargs = {}):
        loss = lossFunc(model, nTrajectories, X, **kwargs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    def train(self, model, nTrajectories, longestTrajectory, X, learningRate = 0.00001, nEpochs = 100, saveDir = "results", **kwargs):
        print("start train!")
        model.train()
        lossFunc = self.loss
        torch.set_num_threads(1)
        optimizer = torch.optim.SGD(model.parameters(), lr = learningRate)
        losses = []
        for epoch in range(nEpochs):
            np.random.shuffle(X)
            results = []
            for batch in range(nBatches):
                loss = self.train_batch(nTrajectories, X[batch * nTrajectories : (batch + 1) *nTrajectories], model, optimizer, lossFunc, kwargs)
                print(f"epoch {epoch + 1}, batch {batch + 1}: loss = {loss:.3f}")

            losses.append(lossFunc(model, nTrajectories,X).item())
            np.savetxt(f"{saveDir}/loss", losses)
            torch.save(model, f"{saveDir}/model_{epoch}.pth")

if __name__ == "__main__":
    setattr(__main__, "Net", SimpleNN)

    model = SimpleNN()
    trainer = trainer()
    nTrajectories = 1000
    nBatches = 1
    longestTrajectory = 0
    trajectoryDir = "directory/of/trajectories"
  
    X = []
    times = []
    for i in range(int(nTrajectories * nBatches)):
        d = pd.read_csv(f"{trajectoryDir}/trajectory{i}")
        X.append(torch.Tensor(list(zip(list(d.x),list(d.y),list(d.z)))))
        times.append(len(d))
      
    trainer.train(model,nTrajectories,longestTrajectory,X)
