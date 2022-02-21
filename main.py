""" Generate fiber forming sequences using the ML model """

import numpy as np
import torch

############################## ML predict fiber ##############################


OUTPUT_SEQ_LEN       = 1
INPUT_DIM            = 20
OUTPUT_DIM           = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cal_feature_matrix(data, max_peplen, res_dict):
    """Convert peptide seqs into a feature matrix
    Accepts python array or list 
    """
    data_feature = np.zeros((len(data), max_peplen, len(res_dict)))
    for i,seq in enumerate(data):
        for j,res in enumerate(seq[:max_peplen]):
            data_feature[i,j,res_dict[res]] = 1
    return data_feature


def featurization(seqs):
    """ encodes each residue into a 1-hot vector """
    # Residue Dictionary
    residues = ['G', 'A', 'V', 'S', 'T', 'L', 'I', 'M', 'P', 'F', 'Y', 'W', 'N', 'Q', 'H', 'K', 'R', 'E', 'D', 'C']
    res_dict = {}
    for i,r in enumerate(residues):
        res_dict[r] = i
    max_peplen = 10
    X = cal_feature_matrix(seqs, max_peplen, res_dict)
    X = X.astype(float)
    return X, seqs


class Model(torch.nn.Module):
    """ ML model
    """
    def __init__(self):
        super().__init__()
        dropout = 0.4
        self.hidden_size = 200
        self.num_layers = 1
        input_size = INPUT_DIM
        self.lstm = torch.nn.LSTM(input_size, self.hidden_size, self.num_layers, dropout=0, batch_first=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.hidden_size,OUTPUT_DIM),
            torch.nn.Sigmoid()
            )
        self.num_params = 0
        self.num_params += sum(p.numel() for p in self.lstm.parameters() if p.requires_grad)
        self.num_params += sum(p.numel() for p in self.fc.parameters() if p.requires_grad)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers,len(x),self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers,len(x),self.hidden_size).to(device)
        out, (h, c) = self.lstm(x, (h0, c0))
        z = self.fc(out[:,-1,:])
        return z


def ml_predict(seqs):
    """ seqs is a list/array of peptide sequence of the PA. 
    e.g. seq is VVAAEE for the PA C16V2A2E2 """

    f = featurization(seqs)
    X, seqs = f[0], f[1]
    checkpoint = torch.load('model.tar')
    model = Model().to(device)
    model.load_state_dict(checkpoint['best_model_state_dict'])
    model.eval()
    x = torch.tensor(X).float()
    z = model(x).detach().numpy()
    
    return seqs, z.reshape(-1)



#########################################################################################


def main():
    # does what you want to do

    seqs = [
    'VAEVAE', 'VVVVEE', 'AVVVEE', 'EVAAVE', 'WVKAK', 'AWKK', 'YLGSRK', 'IISGKK', 
    'VSVMDD', 'HIVRR', 'EVEAEE', 'EVEEVE', 'LSLDDD', 'DVLDD', 'VILLRK']
    _, z = ml_predict(seqs)
    
    print(np.array(list(zip(seqs,z))))



main()






