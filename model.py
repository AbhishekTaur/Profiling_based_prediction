import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init


class RNN_model(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, dtype):
        super(RNN_model, self).__init__()
        self.dtype = dtype
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.w1 = torch.FloatTensor(input_size, hidden_size).type(dtype)
        init.normal_(self.w1, 0.0, 0.4)
        self.w1 = Variable(self.w1, requires_grad=True)
        self.w2 = torch.FloatTensor(hidden_size, output_size).type(dtype)
        init.normal_(self.w2, 0.0, 0.3)
        self.w2 = Variable(self.w2, requires_grad=True)

    def forward(self, input, context_state, w1, w2):
        xh = torch.cat((input, context_state), 1)
        context_state = torch.tanh(xh.mm(w1))
        out = context_state.mm(w2)
        return out, context_state

