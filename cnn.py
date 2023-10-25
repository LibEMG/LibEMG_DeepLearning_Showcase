import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random

#------------------------------------------------#
#             Make it repeatable                 #
#------------------------------------------------#
def fix_random_seed(seed_value, use_cuda=True):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False

#------------------------------------------------#
#            Interfacing with data               #
#------------------------------------------------#
# we require a class for our dataset that has the windows and classes saved
# it needs to have a __getitem__ method that returns the data and label for that id.
# it needs to have a __len__     method that returns the number of samples in the dataset.
class DL_input_data(Dataset):
    def __init__(self, windows, classes):
        self.data = torch.tensor(windows, dtype=torch.float32)
        self.classes = torch.tensor(classes, dtype=torch.long)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data[idx]
        label = self.classes[idx]
        return data, label

    def __len__(self):
        return self.data.shape[0]

def make_data_loader(windows, classes, batch_size=64):
    # first we make the object that holds the data
    obj = DL_input_data(windows, classes)
    # and now we make a dataloader with that object
    dl = DataLoader(obj,
    batch_size=batch_size,
    shuffle=True,
    collate_fn = collate_fn)
    return dl

def collate_fn(batch):
    # this function is used internally by the dataloader (see line 46)
    # it describes how we stitch together the examples into a batch
    signals, labels = [], []
    for signal, label in batch:
        # concat signals onto list signals
        signals += [signal]
        labels += [label]
    # convert back to tensors
    signals = torch.stack(signals)
    labels = torch.stack(labels).long()
    return signals, labels


#------------------------------------------------#
#             Deep Learning Model                #
#------------------------------------------------#
# we require having forward, fit, predict, and predict_proba methods to interface with the 
# EMGClassifier class. Everything else is extra.
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self, n_output, n_channels, n_samples, n_filters=256):
        super().__init__()
        # let's have 3 convolutional layers that taper off
        l0_filters = n_channels
        l1_filters = n_filters
        l2_filters = n_filters // 2
        l3_filters = n_filters // 4
        # let's manually setup those layers
        # simple layer 1
        self.conv1 = nn.Conv1d(l0_filters, l1_filters, kernel_size=5)
        self.bn1   = nn.BatchNorm1d(l1_filters)
        # simple layer 2
        self.conv2 = nn.Conv1d(l1_filters, l2_filters, kernel_size=5)
        self.bn2   = nn.BatchNorm1d(l2_filters)
        # simple layer 3
        self.conv3 = nn.Conv1d(l2_filters, l3_filters, kernel_size=5)
        self.bn3   = nn.BatchNorm1d(l3_filters)
        # and we need an activation function:
        self.act = nn.ReLU()
        # now we need to figure out how many neurons we have at the linear layer
        # we can use an example input of the correct shape to find the number of neurons
        example_input = torch.zeros((1, n_channels, n_samples),dtype=torch.float32)
        conv_output   = self.conv_only(example_input)
        size_after_conv = conv_output.view(-1).shape[0]
        # now we can define a linear layer that brings us to the number of classes
        self.output_layer = nn.Linear(size_after_conv, n_output)
        # and for predict_proba we need a softmax function:
        self.softmax = nn.Softmax(dim=1)
        

    def conv_only(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)
        return x

    def forward(self, x):
        x = self.conv_only(x)
        x = x.view(x.shape[0],-1)
        x = self.act(x)
        x = self.output_layer(x)
        return self.softmax(x)

    def fit(self, dataloader_dictionary, learning_rate=1e-3, num_epochs=100, verbose=True):
        # what device should we use (GPU if available)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # get the optimizer and loss function ready
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        loss_function = nn.CrossEntropyLoss()
        # setup a place to log training metrics
        self.log = {"training_loss":[],
                    "validation_loss": [],
                    "training_accuracy": [],
                    "validation_accuracy": []} 
        # now start the training
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