import time
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer


class PyTorchDescriptionExperiment(object):

    class _Dataset(Dataset):
        
        def __init__(self, X, Y):
            if Y.shape[0] and X.shape[0] != Y.shape[0]:
                raise ValueError('X and Y must be same len')
            self._X = X
            self._Y = Y
            self._sparse_size = torch.Size([1, X.shape[1]]) 
        
        def __len__(self):
            return self._X.shape[0]

        def __getitem__(self, i):
            X = self._X[i]
            if self._Y.any():
                Y = self._Y[i]
            else:
                Y = None
            indices_size = X.indices.shape[0]
            indices = np.append(np.zeros([indices_size, 1]), X.indices.reshape(indices_size, 1), axis=1)
            if indices.shape[0]:
                i = torch.from_numpy(indices).type(torch.LongTensor)
                v = torch.from_numpy(X.data).type(torch.FloatTensor)
                data = torch.sparse.FloatTensor(i.t(), v, self._sparse_size).to_dense()
            else:
                data = torch.sparse.FloatTensor(self._sparse_size).to_dense()
            return {'data': data, 'target': Y}

    class _Net(torch.nn.Module):
        def __init__(self, D_in, H, D_out):
            super().__init__()
            self.linear1 = torch.nn.Linear(D_in, H)
            self.linear2 = torch.nn.Linear(H, D_out)

        def forward(self, x):
            h_relu = self.linear1(x).clamp(min=0)
            y_pred = self.linear2(h_relu)
            return y_pred

    def __init__(self, reader, train, test, x_get, y_get, batch_size, max_samples, max_predictions, print_counter):
        self._reader = reader
        self._train = train
        self._test = test
        self._x_get = x_get
        self._y_get = y_get
        self._batch_size = batch_size
        self._max_samples = max_samples
        self._max_predictions = max_predictions
        self._print_counter = print_counter
        self._tf = TfidfVectorizer()
        self.model = None
        self._train_dataset = None
        self._train_loader = None
    
    def prepare_train(self):
        train = self._reader(self._train)
        if self._max_samples:
            x = self._x_get(train)[:self._max_samples]
            y = self._y_get(train)[:self._max_samples]
        else:
            x = self._x_get(train)
            y = self._y_get(train)
        x_trans = self._tf.fit_transform(x)
        self._x_train_shapes = x_trans.shape
        print("Train dataset shape: %s:%s" % self._x_train_shapes)
        self._train_dataset = self._Dataset(x_trans, y)
        self._train_loader = DataLoader(self._train_dataset, batch_size=self._batch_size)
    
    def train(self, hidden):
        self.model = self._Net(self._x_train_shapes[1], hidden, 1)
        criterion = torch.nn.MSELoss(size_average=False)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4)
        i = [0, time.time()]
        for t in self._train_loader:
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = self.model(Variable(t['data']))

            # Compute and print loss
            loss = criterion(y_pred, Variable(t['target']).type(torch.FloatTensor))
            curr_loss = loss.data[0] 
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i[0] = i[0] + 1
            if i[0] == self._print_counter:
                print(curr_loss)
                print('Time: %s s' % (time.time() - i[1]))
                i[0] = 0
                i[1] = time.time()
    
    def predict(self):
        predictions = []
        test = self._reader(self._test)
        x = self._x_get(test)
        if self._max_predictions:
            x_test = self._tf.transform(x)[:self._max_predictions]
        else:
            x_test = self._tf.transform(x)
        self._x_test_shapes = x_test.shape
        print("Test dataset shape: %s:%s" % self._x_test_shapes)
        self._train_dataset = self._Dataset(x_test, np.array([]))
        for t in self._train_dataset:
            predictions.append(
                self.model(Variable(t['data'])).data.numpy()[0][0]
            )
        return predictions


experiment = PyTorchDescriptionExperiment(
    reader=pd.read_csv,
    train='../data/50K_1K_R_train.csv',
    test='../data/50K_1K_R_test.csv',
    x_get=lambda x: x['item_description'].values,
    y_get=lambda x: np.log1p(x['price'].values),
    batch_size=10,
    max_samples=1000,
    max_predictions=10,
    print_counter=50
)

experiment.prepare_train()
experiment.train(1000)
for i in experiment.predict():
    print(i)