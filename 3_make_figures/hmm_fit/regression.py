#!/bin/python

#%%

import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from matplotlib import use
# use('agg')


class Net (nn.Module):
    '''
    Network model
    '''
    def __init__(self, input_dims, output_dims, weight_decay=1.e-3):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims[0]

        # define the feed-forward architecture
        self.layers = nn.Sequential(
                nn.Linear(*self.input_dims, self.output_dims),
                nn.Softmax(),
            )
        # the algorithm that perform gradient-based optimisation
        self.optimizer = optim.Adam(self.parameters(), lr=.001, weight_decay=weight_decay)
        # this is to run the network optimisation on a GPU, if available
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, x):
        '''
        Forward pass: this MUST be there.
        It computes the output of the network given the input x
        '''
        return self.layers(x)

    def save (self, path):
        '''
        save the parameters of the network
        '''
        T.save(self.state_dict(), path)

    def load (self, path):
        '''
        load the parameters of the network from file
        '''
        self.load_state_dict(T.load(path, map_location=self.device))


class Regression (object):
    '''
    class to train and test the network
    '''

    def __init__ (self, X0, y0, models_dir=".", model_name="network", loss=F.mse_loss, weight_decay=1.e-2):
        '''
        Gets as arguments a template of the input and a template of an output
        It uses the dimensions of these to define the network
        '''
        self.input_dims = X0.shape
        if len(y0.shape) > 0:
            self.output_dims = y0.shape
        else:
            self.output_dims = (1,)
        self.net = Net (self.input_dims, self.output_dims, weight_decay=weight_decay)
        self.models_dir = models_dir
        self.loss = loss


    @property
    def device(self):
        return self.net.device

    @property
    def predict (self):
        def _predict (X):
            _x = T.tensor(X, dtype=T.float).to(self.device)
            return self.net.forward(_x).cpu().detach().numpy().squeeze()
        return _predict

    def train (self, X_train, y_train, batch_size=50, n_epochs=100,
                     X_test=None, y_test=None, monitor=False,
        ):
        '''
        Train the neural network model
        '''
        X = T.tensor(X_train, dtype=T.float)
        y = T.tensor(y_train, dtype=T.float).view(-1,*self.output_dims)
        if monitor:
            train_losses = np.zeros(n_epochs)
            test_losses = np.zeros(n_epochs)
            try:
                X_t = T.tensor(X_test, dtype=T.float)
                y_t = T.tensor(y_test, dtype=T.float).view(-1,*self.output_dims)
            except:
                raise ValueError("Check X_test and y_test")

        # every epoch is a sweep through all the training data
        for epoch in range(n_epochs):
            # every epoch is divided into batches, over which the loss
            # gradient is estimated
            # (consider shuffling the training data at the
            # beginning of every epoch)
            idx = T.randperm(X.shape[0])
            X = X[idx]
            y = y[idx]
            for i in range(0, len(X), batch_size):
                # define the batch of data
                X_batch = X[i:i+batch_size].to(self.device)
                y_batch = y[i:i+batch_size].view(-1,*self.output_dims).to(self.device)
                self.net.optimizer.zero_grad()     # (required for backpropagation)
                output = self.net.forward(X_batch) # predict
                loss = self.loss(output, y_batch) # least-square optimisation
                loss.backward()                    # calculate the gradient of the loss wrt parameters
                self.net.optimizer.step()          # optimisation step

            # monitor training and testing loss
            if monitor:
                train_loss = self.test(X, y)
                _message = f'Epoch: {epoch:3d}.\tTrain loss: {train_loss:.4E}'
                test_loss = self.test(X_t, y_t)
                _message += f'\t\tTest loss: {test_loss:.4E}'
                train_losses[epoch] = train_loss
                test_losses[epoch] = test_loss
                print(_message)

        self.save()
        if monitor:
            return train_losses, test_losses


    def test (self, X_test, y_test):

        if not T.is_tensor(X_test):
            X_test = T.tensor(X_test, dtype=T.float)
        if not T.is_tensor(y_test):
            y_test = T.tensor(y_test, dtype=T.float)

        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)

        with T.no_grad(): 
            loss = self.loss(self.net.forward(X_test), y_test)
        return loss

    def score (self, X_test, y_test, y_mean=None):

        y_pred = self.predict(X_test)
        res_ = y_pred - y_test
        if y_mean is None:
            y_mean = np.mean(y_test, axis=0)
        res_m = y_pred - y_mean[None]

        return 1. - np.mean(res_**2) / np.mean(res_m**2)

    def save (self, model_name="network"):
        self.net.save(os.path.join(self.models_dir, model_name))

    def load (self, model_name="network"):
        self.net.load(os.path.join(self.models_dir, model_name))


    def test_plot (self, X_test, y_test, filename="prediction.svg",
            fig=None, ax=None, **kwargs):

        y_pred = self.predict(X_test)

        close = False
        if fig is None or ax is None:
            fig, ax = plt.subplots()
            close = True

        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        if len(y_pred.shape) == 1:
            ax.scatter(y_test, y_pred, s=.5, **kwargs)
        else:
            for u,v in zip(y_test.T, y_pred.T):
                ax.scatter(u, v, s=.5, **kwargs)
        _min = min(y_test.min(),y_pred.min())
        _max = max(y_test.max(),y_pred.max())
        ax.plot([_min,_max],[_min,_max], c='k', ls='--')
        ax.set_title(f"R2 score = {self.score(X_test, y_test):.4f}")
        plt.savefig(filename, bbox_inches='tight')
        if close:
            plt.close(fig)



#%%
if __name__ == "__main__":


    #############################################################
    #
    #                   DATA FOR REGRESSION
    #
    #############################################################

    output_dir = "test_outputs_2d"
    os.makedirs(output_dir, exist_ok="True")

    def target_function (x):
        if len(x.shape) == 1:
            x = x[None]
        # return x[:,0] + x[:,1]**2
        return np.vstack((x[:,0] + x[:,1]**2, x[:,0]**3 - x[:,1])).T

    N = 1000
    X = 2. * (2. * np.random.rand(N,2).astype(float) - 1. )
    y_true = target_function(X)
    y = y_true + 0.2 * np.random.randn(*y_true.shape)


    #############################################################
    #
    #                REGRESSION / CLASSIFICATION
    #
    #############################################################
    
    from sklearn.model_selection import train_test_split


    # monitor training, and check for overfitting
    n_epochs = 30
    n_exp = 10
    train_losses = []
    test_losses = []
    for i in range(n_exp):
        print("\nExperiment ", i)
        model = Regression(X[0], y[0], models_dir=output_dir)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        train_loss, test_loss = model.train(X_train, y_train,
                    n_epochs=n_epochs, X_test=X_test, y_test=y_test, monitor=True)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    train_losses = np.array(train_losses)
    test_losses = np.array(test_losses)

    np.save(os.path.join(output_dir,"losses_train.npy"), train_losses)
    np.save(os.path.join(output_dir,"losses_test.npy"), test_losses)

    train_mid = np.median(train_losses, axis=0)
    test_mid = np.median(test_losses, axis=0)

    # plot training curves
    n_stopping = 20
    fig, ax = plt.subplots()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    train_low = np.percentile(train_losses, 10, axis=0)
    train_high = np.percentile(train_losses, 90, axis=0)
    test_low = np.percentile(test_losses, 10, axis=0)
    test_high = np.percentile(test_losses, 90, axis=0)

    ax.fill_between(np.arange(n_epochs), train_low, train_high, color='C0', alpha=0.2)
    ax.fill_between(np.arange(n_epochs), test_low, test_high, color='C1', alpha=0.2)

    ax.plot(train_mid, color='C0', lw=2, label="train loss")
    ax.plot(test_mid, color='C1', lw=2, label="test loss")
    ax.plot([n_stopping,n_stopping],[np.min(train_low), np.max(test_high)], c='k', ls='--')
    ax.legend()
    fig.savefig(os.path.join(output_dir,"training.svg"), bbox_inches='tight')
    plt.close(fig)


    # fit again, stopping early
    model = Regression(X[0], y[0], models_dir=output_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(f"Training for {n_stopping} epochs")
    model.train(X_train, y_train, n_epochs=n_stopping)

    y_pred = model.predict(X_test)

    fig, ax = plt.subplots()
    model.test_plot(X_test, y_test, filename=os.path.join(output_dir,"prediction.svg"))
    plt.close(fig)