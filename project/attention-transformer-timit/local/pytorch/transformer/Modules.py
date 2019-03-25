import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

__author__ = "Yu-Hsiang Huang"

class Linear(nn.Module):
    ''' Simple Linear layer with xavier init '''
    def __init__(self, d_in, d_out, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)

class Bottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0]*size[1], -1))
        return out.view(size[0], size[1], -1)

class BottleLinear(Bottle, Linear):
    ''' Perform the reshape routine before and after a linear projection '''
    pass

class LayerNormalization(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out

class BatchBottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(BatchBottle, self).forward(input)
        size = input.size()[1:]
        out = super(BatchBottle, self).forward(input.view(-1, size[0]*size[1]))
        return out.view(-1, size[0], size[1])

class BottleLayerNormalization(BatchBottle, LayerNormalization):
    ''' Perform the reshape routine before and after a layer normalization'''
    pass

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d_model, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, attn_mask=None):

        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

        if attn_mask is not None:

            assert attn_mask.size() == attn.size(), \
                    'Attention mask shape {} mismatch ' \
                    'with Attention logit tensor shape ' \
                    '{}.'.format(attn_mask.size(), attn.size())

            attn.data.masked_fill_(attn_mask, -float('inf'))
            
        attn = nn.functional.softmax(attn, dim=2)
        #incase condition of all -inf, in this condition, softmax will return nan and cause error
        attn.data.masked_fill_(attn_mask, 0)

        #self.attn_norm_grad = self.get_norm_grad(attn)
        #attn.register_hook(self.grad_hook)

        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn

    def get_norm_grad(self, attn):
        size = attn.size()
        T = torch.FloatTensor(range(size[2])).repeat(size[0], size[1], 1)
        if attn.is_cuda:
            T = T.cuda()
        average = (attn * T).sum(2)
        average = average.view(size[0], size[1], 1).repeat(1, 1, size[2])

        scale = 0.03
        norm_grad = ((T - average)/size[1]) ** 2 * scale
        return norm_grad

    def grad_hook(self, grad):
        #torch.set_printoptions(precision=3, threshold=None, edgeitems=20, linewidth=None, profile=None)
        #print(self.attn_norm_grad)
        #exit(0)
        grad += self.attn_norm_grad