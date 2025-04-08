# Author: Kshitij Kayastha
# Date: Feb 3, 2025


import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression



class Model(ABC):
	def __init__(self):
		pass

	def metrics(self, X, y):
		acc = np.mean(self.predict(X)==y)

		pred = self.predict_proba(X)[:,1]
		fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
		auc = metrics.auc(fpr, tpr)

		return acc, auc
 
	@abstractmethod
	def train(self, X, y):
		pass


class LR(Model):
	def __init__(self):
		super(LR, self).__init__()

	def train(self, X, y):
		model = LogisticRegression().fit(X, y)
		self.model = model

		self.weights = torch.from_numpy(model.coef_[0]).float()
		self.bias = torch.tensor(model.intercept_[0]).float()

	def torch_model(self, x):
		return torch.nn.Sigmoid()(torch.matmul(self.weights, x) + self.bias)[0]

	def predict(self, x):
		return self.model.predict(x)

	def predict_proba(self, x):
		return self.model.predict_proba(x)


class NN(Model):
	def __init__(self, n_features):
		torch.manual_seed(0)
		super(NN, self).__init__()
		self.model = nn.Sequential(
		  nn.Linear(n_features, 50),
		  nn.ReLU(),
		  nn.Linear(50, 100),
		  nn.ReLU(),
		  nn.Linear(100, 200),
		  nn.ReLU(),
		  nn.Linear(200, 1),
		  nn.Sigmoid()
		  )
	
	def torch_model(self,x):
		return self.model(x)[0]

	def train(self, X_train, y_train, verbose=0):
		torch.manual_seed(0)
		X_train = torch.from_numpy(X_train).float()
		y_train = torch.from_numpy(y_train).float()

		loss_fn = nn.BCELoss()
		optimizer = torch.optim.Adam(self.model.parameters())

		epochs = 100
		for epoch in range(epochs):
			self.model.train()
			optimizer.zero_grad()

			y_pred = self.model(X_train)
			loss = loss_fn(y_pred[:,0], y_train)
			if verbose: print(f'Epoch {epoch}: train loss: {loss.item()}')

			loss.backward()
			optimizer.step()

	def predict_proba(self, X: np.ndarray):
		X = torch.from_numpy(np.array(X)).float()
		class1_probs = self.model(X).detach().numpy()
		class0_probs = 1-class1_probs
		return np.hstack((class0_probs,class1_probs))

	def predict(self, X):
		return np.argmax(self.predict_proba(X), axis=1)


class LogReg(Model):
    def __init__(self, num_feat):
        torch.manual_seed(0)
        super(LogReg, self).__init__()
        self.model = nn.Sequential(
		  nn.Linear(num_feat, 1),
		  nn.Sigmoid()
		  )
        
    def forward(self, X):
        return self.model(X)
    
    def torch_model(self,x):
        return self.model(x)[0]

    def train(self, X_train, y_train, verbose=0):
        torch.manual_seed(0)
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()
        
        loss_fn = nn.BCELoss()		
        optimizer = torch.optim.Adam(self.model.parameters())

		# Train model
        epochs = 100
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            
            y_pred = self.model(X_train)
            loss = loss_fn(y_pred[:,0], y_train)
            if verbose: print(f'Epoch {epoch}: train loss: {loss.item()}')
            
            loss.backward()
            optimizer.step()
        return self

    def predict_proba(self, X):
        X = torch.from_numpy(np.array(X)).float()
        class1_probs = self.model(X).detach().numpy()
        class0_probs = 1-class1_probs
        return np.hstack((class0_probs,class1_probs))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
    

class Net0(nn.Module):
    def __init__(self, n):
        super(Net0, self).__init__()
        self.fc1 = nn.Linear(n, 20)
        self.fc2 = nn.Linear(20, 50)
        self.fc3 = nn.Linear(50, 20)
        self.out = nn.Linear(20, 1)
        self.rng = np.random.default_rng(0)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(F.relu(self.out(x)))
        return x

    def predict_proba(self, inp):
        if not isinstance(inp, torch.Tensor):
            inp = torch.tensor(inp, dtype=torch.float32)
        else:
            inp = inp.float()
        out = self.forward(inp).detach()
        out = torch.cat([1 - out, out], axis=-1)
        return out.numpy()
    
    def predict(self, inp):
        
        out = self.predict_proba(inp)
        # y_pred = np.argmax(out, axis=-1)
        if out[0,0] != out[0,1]:
            y_pred = np.argmax(out, axis=-1)
        else:
            i = self.rng.choice([0, 1])
            y_pred = np.array([[0., 1.]])[:, i]
        return y_pred