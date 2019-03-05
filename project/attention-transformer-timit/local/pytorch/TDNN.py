import torch
import torch.nn as nn
import torch.nn.functional as F

class TDNNLayer(nn.Module):
    """docstring for TDNNLayer"""
    def __init__(self, d_input, d_output, index, dropout=0.1):
        super(TDNNLayer, self).__init__()
        if len(index) < 2:
            print('[ERROR] invalid concat index length')
            exit(1)

        self.index = index
        if index[0] < 0:
            self.pad_head = -index[0]
        else:
            self.pad_head = 0

        if index[-1] > 0:
            self.pad_end = index[-1]
        else:
            self.pad_end = 0

        self.proj = nn.Linear(d_input*len(index), d_output, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        frame = x.shape[1]
        pad_data = F.pad(x,(0,0,self.pad_head,self.pad_end),'constant', 0)

        i = self.index[0]
        cat_data = pad_data[:,i+self.pad_head:i+self.pad_head+frame,:]
        for i in self.index[1:]:
            cat_data = torch.cat((cat_data,pad_data[:,i+self.pad_head:i+self.pad_head+frame,:]),2)

        outputs = self.proj(cat_data)
        outputs = self.relu(outputs)
        outputs = self.dropout(outputs)
        return outputs