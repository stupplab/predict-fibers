""" Fiber predicting sequences using the ML model """

import os, time
import copy
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tqdm


################################## ML ARCHITECTURE ################################

class Dataset(torch.utils.data.Dataset):
    def __init__(self,X,y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        return self.X[idx], self.y[idx]



def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, torch.nn.LSTM):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)



class LSTM(torch.nn.Module):
    """ ML model architecture
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        dropout = 0.4
        self.hidden_size = 200
        self.num_layers = 1

        self.lstm = torch.nn.LSTM(input_size, self.hidden_size, self.num_layers, dropout=0, batch_first=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.hidden_size, output_size),
            torch.nn.Sigmoid()
            )
            
        self.num_params = 0
        self.num_params += sum(p.numel() for p in self.lstm.parameters() if p.requires_grad)
        self.num_params += sum(p.numel() for p in self.fc.parameters() if p.requires_grad)

    def forward(self, x):
        
        h0 = torch.zeros(self.num_layers,len(x),self.hidden_size)
        c0 = torch.zeros(self.num_layers,len(x),self.hidden_size)
        
        out, (h, c) = self.lstm(x, (h0, c0))
        z = self.fc(out[:,-1,:])
        return z

###################################################################################

################################## MODEL CLASS ####################################

class Model():
    """ Main class """

    def __init__(self, model_path='model.tar'):
        # parameters
        self.BATCH_SIZE    = 64
        self.N_EPOCHS      = 100
        self.LR            = 1e-3

        self.INPUT_DIM     = 20
        self.OUTPUT_DIM    = 1
        
        self.loss_scale    = 1
        self.model_path    = model_path

        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_class = LSTM


    def one_hot_encode(self, filename):
        """ one-hot encode the letter sequence """
        
        dataframe = pd.read_csv(filename, sep=',', header=None)
        seqs = dataframe.values[:,0]
        
        y = dataframe.values[:,1]
        y = y.reshape(-1,1).astype(float)

        # Residue Dictionary
        residues = ['G', 'A', 'V', 'S', 'T', 'L', 'I', 'M', 'P', 'F', 'Y', 'W', 'N', 'Q', 'H', 'K', 'R', 'E', 'D', 'C']
        res_dict = {}
        for i,r in enumerate(residues):
            res_dict[r] = i
        max_peplen = 10

        X = np.zeros((len(seqs), max_peplen, len(res_dict)))
        for i,seq in enumerate(seqs):
            for j,res in enumerate(seq[:max_peplen]):
                X[i,j,res_dict[res]] = 1
        
        X = X.astype(float)
        
        print(f"Sample Distribution: {len(X)} ({np.sum(y)/len(y)})")

        return X, y, seqs


    def new_model(self):
        self.model = self.model_class(self.INPUT_DIM, self.OUTPUT_DIM).to(self.device)
        self.last_loss = 1000
        self.last_precision = 0
        self.checkpoint = {}

    
    def load_model(self):
        self.checkpoint = torch.load(self.model_path)
        self.model = self.model_class(self.INPUT_DIM, self.OUTPUT_DIM).to(self.device)
        self.model.load_state_dict(self.checkpoint['last_model_state_dict'])
        try:
            self.last_precision = checkpoint['last_precision']
        except:
            self.last_precision = 0
        try:
            self.last_loss = checkpoint['last_loss']
        except:
            self.last_loss = 1000


    def init_loss_function(self):
        self.loss_function = torch.nn.BCELoss(reduction='mean')


    def init_optimizer_function(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LR)


    def train_step(self, train_iterator):
        # set the train mode
        self.model.train()

        # loss of the epoch
        train_loss = []

        for i, (X, y) in enumerate(train_iterator):
            X = X.float()
            y = y.float()

            X = X.to(self.device)

            # update the gradients to zero
            self.optimizer.zero_grad()

            # forward pass
            z = self.model(X)
            
            # loss
            loss = self.loss_function(z, y)
            
            # backward pass
            loss.backward()
            
            train_loss += [loss.item()]
            
            # update the weights
            self.optimizer.step()

        return np.mean(train_loss)



    def test_step(self, test_iterator):
        # set the evaluation mode
        self.model.eval()

        # test loss for the data
        test_loss = []

        # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing
        with torch.no_grad():
            for i, (X, y) in enumerate(test_iterator):
                X = X.float()
                y = y.float()
                
                X = X.to(self.device)

                # forward pass
                z = self.model(X)

                # loss
                loss = self.loss_function(z, y)

                # total loss
                test_loss += [loss.item()]

        return np.mean(test_loss)


    def accuracy(self, X, y):
        """ Accuracy """
    
        self.model.eval()
        z = self.model( torch.tensor(X).float() ).detach().numpy()
        
        correct = y == np.round(z)
        denominator = len(correct)

        if len(correct)==0:
            accu = 0.0
        else:
            accu = 100 * np.sum(correct) / denominator
        
        return accu


    def confusion(self, X, y):
        """ confusion matrix calculation """
        
        self.model.eval()
        z = self.model( torch.tensor(X).float() ).detach().numpy()
        z = np.round(z)

        P = sum(y)
        N = len(y) - P
        TP = sum(y*z)
        FP = sum((1-y)*z)
        TN = sum((1-y)*(1-z))
        FN = sum(y*(1-z))

        return P, N, TP, FP, TN, FN


    def train(self, train_datafile, test_datafile):
        # train the model

        f = self.one_hot_encode(train_datafile)
        X_train, y_train = f[0], f[1]
        f = self.one_hot_encode(test_datafile)
        X_test, y_test = f[0], f[1]

        # optimizer
        self.init_optimizer_function()

        # init loss function
        self.init_loss_function()

        # Create iterators for train and test data
        train_dataset = Dataset(X_train, y_train)
        test_dataset = Dataset(X_test, y_test)
        train_iterator = torch.utils.data.DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=False)
        test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=self.BATCH_SIZE)


        print(f'Num trainable params: {self.model.num_params}, Num train samples: {len(X_train)}')
        accuracy_train = self.accuracy(X_train, y_train)
        accuracy_test = self.accuracy(X_test, y_test)
        print(f'Accuracy - Train: {accuracy_train:.1f}, Test: {accuracy_test:.1f}')
        evaluation = dict(train_loss=[], validation_loss=[], test_loss=[], 
            accuracy_train=[], accuracy_validation=[], accuracy_test=[])
        for e in tqdm.tqdm(range(self.N_EPOCHS)):
            train_loss = self.train_step(train_iterator)
            test_loss  = self.test_step(test_iterator)

            accuracy_train = self.accuracy(X_train, y_train)
            accuracy_test  = self.accuracy(X_test, y_test)
            P, N, TP, FP, TN, FN = self.confusion(X_test, y_test)
            precision = 100 * TP[0] / ( TP[0] + FP[0] )
            print(f'Epoch {e:3d} Loss (Accu): Train {train_loss:.3f}({accuracy_train:.1f}), Test {test_loss:.2f}({accuracy_test:.1f}), Precision {precision:.1f}')

            evaluation['train_loss'] += [train_loss]
            evaluation['test_loss'] += [test_loss]
            evaluation['accuracy_train'] += [accuracy_train]
            evaluation['accuracy_test'] += [accuracy_test]

            # save
            self.checkpoint['last_model_state_dict'] = self.model.state_dict()
            if test_loss < self.last_loss:
                # self.checkpoint['best_model_state_dict'] = copy.deepcopy(self.model.state_dict())
                self.checkpoint['loss'] = test_loss
                self.last_loss = test_loss
            
            if precision > self.last_precision:
                self.checkpoint['best_model_state_dict'] = copy.deepcopy(self.model.state_dict())
                self.checkpoint['precision'] = precision
                self.last_precision = precision

            torch.save(self.checkpoint, self.model_path)
            torch.save(evaluation, 'evaluation.tar')


    def predict(self, X):
        self.model.eval()
        z = self.model( torch.tensor(X).float() ).detach().numpy()
        return z

##################################### LOAD MODEL ##################################
def ml_predict(seqs):
    model = Model(model_path='model.tar')
    model.load_model()
    model.model.load_state_dict(model.checkpoint['best_model_state_dict'])
    residues = ['G', 'A', 'V', 'S', 'T', 'L', 'I', 'M', 'P', 'F', 'Y', 'W', 'N', 'Q', 'H', 'K', 'R', 'E', 'D', 'C']
    res_dict = {}
    for i,r in enumerate(residues):
        res_dict[r] = i
    max_peplen = 10
    X = np.zeros((len(seqs), max_peplen, len(res_dict)))
    for i,seq in enumerate(seqs):
        for j,res in enumerate(seq[:max_peplen]):
            X[i,j,res_dict[res]] = 1
    z = list(np.round(model.predict(X),1).reshape(-1))
    prediction = list(np.array(['nonfiber','fiber'])[np.round(z).astype(int)].reshape(-1))
    del model
    return np.array(list(zip(seqs,prediction,z)))

###################################################################################

################################## USER ADD SEQS HERE #############################

seqs = [
    'VVAAEE', 'VVVAAAEEE',
    'VAEVAE', 'VVVVEE', 'AVVVEE', 'EVAAVE', 'WVKAK', 'AWKK', 'YLGSRK', 'IISGKK', 
    'VSVMDD', 'HIVRR', 'EVEAEE', 'EVEEVE', 'LSLDDD', 'DVLDD', 'VILLRK'
    ]


print(ml_predict(seqs))

###################################################################################






