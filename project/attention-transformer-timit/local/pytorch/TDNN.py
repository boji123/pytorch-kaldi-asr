import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class ConcatLayer(nn.Module):
    def __init__(self, index=[0]):
        super(ConcatLayer, self).__init__()
        self.index = index
        if index[0] < 0:
            self.pad_head = -index[0]
        else:
            self.pad_head = 0

        if index[-1] > 0:
            self.pad_end = index[-1]
        else:
            self.pad_end = 0

    def forward(self, x):
        bacth = x.shape[0]
        frame = x.shape[1]
        dim = x.shape[2]

        pad_data = F.pad(x,(0,0,self.pad_head,self.pad_end),'constant', 0)
        cat_data_list = [pad_data[:,self.index[i]+self.pad_head:self.index[i]+self.pad_head+frame,:] for i in range(len(self.index))]
        cat_data = torch.cat(cat_data_list,2)
        return cat_data


class TDNNLayer(nn.Module):
    """docstring for TDNNLayer"""
    def __init__(self, d_input, d_output, index, dropout=0.1):
        super(TDNNLayer, self).__init__()
        self.concat = ConcatLayer(index)
        self.proj = nn.Linear(d_input*len(index), d_output, bias=True)
        init.xavier_normal_(self.proj.weight)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        cat_data = self.concat(x)
        outputs = self.proj(cat_data)
        outputs = self.relu(outputs)
        outputs = self.dropout(outputs)
        return outputs

class LDALayer(nn.Module):
    def __init__(self, LDA_mat):
        super(LDALayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(LDA_mat[:,:-1]).transpose(0, 1), requires_grad=False)
        self.bias = nn.Parameter(torch.FloatTensor(LDA_mat[:,-1]), requires_grad=False)
    def forward(self, x):
        output = x.matmul(self.weight) + self.bias
        return output


class vFSMNLayer(nn.Module):
    def __init__(self, dim, index, dropout=0):
        super(vFSMNLayer, self).__init__()
        self.concat = ConcatLayer(index)
        self.memory = nn.Parameter(torch.FloatTensor(len(index) * dim), requires_grad=True)
        self.proj = nn.Linear(dim*2, dim, bias=True)
        init.xavier_normal_(self.proj.weight)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        size = x.size()
        #vectorized memory
        outputs = self.concat(x)
        outputs = self.memory.mul(outputs)
        outputs = outputs.view(size[0],size[1],-1,size[2])
        outputs = outputs.sum(2)
        #product
        outputs = torch.cat([outputs,x],2)
        outputs = self.proj(outputs)
        outputs = self.relu(outputs)
        outputs = self.dropout(outputs)
        return outputs


class ConvLayer(nn.Module):
    def __init__(self, tstep, fstep):
        super(ConvLayer, self).__init__()
        self.tstep = tstep
        self.fstep = fstep
        self.conv = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(tstep,fstep))
        init.xavier_normal_(self.conv.weight)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = F.pad(x,(self.fstep-1,0,self.tstep-1,0),'constant', 0)
        output = output.view(output.size()[0],1,output.size()[1],output.size()[2])
        #input batch * 1(channel) * pad+len * pad+dim
        #filter 1(num) * 1(channel) * tstep * fstep
        output = self.conv(output)
        output = output.squeeze()
        output = self.relu(output)
        return output