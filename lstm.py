import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random

#------------------------------------------------#
#             Deep Learning Model                #
#------------------------------------------------#
# we require having forward, fit, predict, and predict_proba methods to interface with the 
# EMGClassifier class. Everything else is extra.
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTM(nn.Module):
    def __init__(self, n_output, n_features, hidden_layers=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_layers, num_layers=2, batch_first=True)
        self.output_layer = nn.Linear(hidden_layers, n_output)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.output_layer(x)
        return self.softmax(x)

    def fit(self, dataloader_dictionary, learning_rate=1e-3, num_epochs=100, verbose=True):
        # what device should we use (GPU if available)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # get the optimizer and loss function ready
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        loss_function = nn.CrossEntropyLoss()
        self.log = {"training_loss":[],
            "validation_loss": [],
            "training_accuracy": [],
            "validation_accuracy": []} 
        
        for epoch in range(num_epochs):
            #training set
            self.train()
            for data, labels in dataloader_dictionary["training_dataloader"]:
                optimizer.zero_grad()
                data = data.to(device)
                labels = labels.to(device)
                output = self.forward(data)
                loss = loss_function(output, labels)
                loss.backward()
                optimizer.step()
                acc = sum(torch.argmax(output,1) == labels)/labels.shape[0]
                # log it
                self.log["training_loss"] += [(epoch, loss.item())]
                self.log["training_accuracy"] += [(epoch, acc)]
            # validation set
            self.eval()
            for data, labels in dataloader_dictionary["validation_dataloader"]:
                data = data.to(device)
                labels = labels.to(device)
                output = self.forward(data)
                loss = loss_function(output, labels)
                acc = sum(torch.argmax(output,1) == labels)/labels.shape[0]
                # log it
                self.log["validation_loss"] += [(epoch, loss.item())]
                self.log["validation_accuracy"] += [(epoch, acc)]
            if verbose:
                epoch_trloss = np.mean([i[1] for i in self.log['training_loss'] if i[0]==epoch])
                epoch_tracc  = np.mean([i[1] for i in self.log['training_accuracy'] if i[0]==epoch])
                epoch_valoss = np.mean([i[1] for i in self.log['validation_loss'] if i[0]==epoch])
                epoch_vaacc  = np.mean([i[1] for i in self.log['validation_accuracy'] if i[0]==epoch])
                print(f"{epoch}: trloss:{epoch_trloss:.2f}  tracc:{epoch_tracc:.2f}  valoss:{epoch_valoss:.2f}  vaacc:{epoch_vaacc:.2f}")
        self.eval()

    def predict(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        y = self.forward(x)
        predictions = torch.argmax(y, dim=1)
        return predictions.cpu().detach().numpy()

    def predict_proba(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        y = self.forward(x)
        return y.cpu().detach().numpy()