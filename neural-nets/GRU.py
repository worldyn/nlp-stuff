import torch
from torch import nn
import numpy as np
from torch.autograd import Variable

class GRUCellV2(nn.Module):
    """
    GRU cell implementation
    """
    def __init__(self, input_size, hidden_size, activation=torch.tanh):
        """
        Initializes a GRU cell

        :param      input_size:      The size of the input layer
        :type       input_size:      int
        :param      hidden_size:     The size of the hidden layer
        :type       hidden_size:     int
        :param      activation:      The activation function for a new gate
        :type       activation:      callable
        """
        super(GRUCellV2, self).__init__()
        self.activation = activation
        self.hidden_size = hidden_size
        self.input_size = input_size

        # initialize weights by sampling from a uniform distribution between -K and K
        K = 1 / np.sqrt(hidden_size)
        
        # weights
        self.w_ih = nn.Parameter(torch.rand(3 * hidden_size, input_size) * 2 * K - K)
        self.w_hh = nn.Parameter(torch.rand(3 * hidden_size, hidden_size) * 2 * K - K)
        self.b_ih = nn.Parameter(torch.rand(3 * hidden_size) * 2 * K - K)
        self.b_hh = nn.Parameter(torch.rand(3 * hidden_size) * 2 * K - K)

        
    def forward(self, x, h):
        """
        Performs a forward pass through a GRU cell


        Returns the current hidden state h_t for every datapoint in batch.
        
        :param      x:    an element x_t in a sequence
        :type       x:    torch.Tensor
        :param      h:    previous hidden state h_{t-1}
        :type       h:    torch.Tensor
        """
        #
        # YOUR CODE HERE
        #
        # split chunks to weight matrices matmuled with x
        B = x.shape[0]
        W_ir_x, W_iz_x, W_in_x = torch.chunk(torch.matmul(self.w_ih, x.T), 3)
        W_hr_h, W_hz_h, W_hn_h = torch.chunk(torch.matmul(self.w_hh, h.T), 3)
        # extract biases and repeat for batch computations
        b_ir, b_iz, b_in = list(map(
            lambda b: b.reshape(1,self.hidden_size).T.repeat(1,B), 
            torch.chunk(self.b_ih, 3)
        ))
        b_hr, b_hz, b_hn = list(map(
            lambda b: b.reshape(1,self.hidden_size).T.repeat(1,B), 
            torch.chunk(self.b_hh, 3) 
        )) 
        
        # GRU operations 
        r = torch.sigmoid(W_ir_x + b_ir + W_hr_h + b_hr)
        z = torch.sigmoid(W_iz_x + b_iz + W_hz_h + b_hz)
        h_in = self.activation(W_in_x + b_in + r * (W_hn_h + b_hn))
        h_t = (1-z) * h_in + z * h.T # new hidden h_t
        return h_t.T


class GRU2(nn.Module):
    """
    GRU network implementation
    """
    def __init__(self, input_size, hidden_size, bias=True, activation=torch.tanh, bidirectional=False):
        super(GRU2, self).__init__()
        self.bidirectional = bidirectional
        self.fw = GRUCellV2(input_size, hidden_size, activation=activation) # forward cell
        if self.bidirectional:
            self.bw = GRUCellV2(input_size, hidden_size, activation=activation) # backward cell
        self.hidden_size = hidden_size
        
    def forward(self, x):
        """
        Performs a forward pass through the whole GRU network, consisting of a number of GRU cells.
        Takes as input a 3D tensor `x` of dimensionality (B, T, D),
        where B is the batch size;
              T is the sequence length (if sequences have different lengths, they should be padded before being inputted to forward)
              D is the dimensionality of each element in the sequence, e.g. word vector dimensionality

        The method returns a 3-tuple of (outputs, h_fw, h_bw), if self.bidirectional is True,
                           a 2-tuple of (outputs, h_fw), otherwise
        `outputs` is a tensor containing the output features h_t for each t in each sequence (the same as in PyTorch native GRU class);
                  NOTE: if bidirectional is True, then it should contain a concatenation of hidden states of forward and backward cells for each sequence element.
        `h_fw` is the last hidden state of the forward cell for each sequence, i.e. when t = length of the sequence;
        `h_bw` is the last hidden state of the backward cell for each sequence, i.e. when t = 0 (because the backward cell processes a sequence backwards)
        
        :param      x:    a batch of sequences of dimensionality (B, T, D)
        :type       x:    torch.Tensor
        """
        #
        # YOUR CODE HERE
        B, T, D = x.shape

        # insert a constant vector first
        h_fw = Variable(torch.zeros(B,self.hidden_size), requires_grad=False)
        # init gru outputs
        outs_fw = torch.empty(T,B,self.hidden_size)

        # same for bidir
        if self.bidirectional:
            h_bw = Variable(torch.zeros(B,self.hidden_size), requires_grad=False)
            outs_bw = torch.empty(T,B,self.hidden_size)
        
        # loop through GRUs
        for t_fw in range(T):
            # self.fw, self.bw: GRUs
            x_fw = x[:,t_fw,:] # input x at time step t forward
            h_fw = self.fw(x_fw, h_fw) 
            outs_fw[t_fw] = h_fw # set to new current hid
            if self.bidirectional:
                t_bw = T - 1 - t_fw # time step backwards
                x_bw = x[:,t_bw,:] # x at time step t backward
                h_bw = self.bw(x_bw, h_bw) 
                outs_bw[t_bw] = h_bw # set no new current hid

        # concat if bi-dir
        if self.bidirectional:
            outs_cat = torch.cat((outs_fw, outs_bw), dim=2)
            return outs_cat, h_fw, h_bw
        return outs_fw, h_fw

def is_identical(a, b):
    return "Yes" if np.all(np.abs(a - b) < 1e-6) else "No"


if __name__ == '__main__':
    torch.manual_seed(100304343)
    x = torch.randn(5, 3, 10)
    gru = nn.GRU(10, 20, bidirectional=False, batch_first=True)
    outputs, h = gru(x)

    torch.manual_seed(100304343)
    x = torch.randn(5, 3, 10)
    gru2 = GRU2(10, 20, bidirectional=False)
    outputs, h_fw = gru2(x)
    
    print("Checking the unidirectional GRU implementation")
    print("Same hidden states of the forward cell?\t\t{}".format(
        is_identical(h[0].detach().numpy(), h_fw.detach().numpy())
    ))

    torch.manual_seed(100304343)
    x = torch.randn(5, 3, 10)
    gru = GRU2(10, 20, bidirectional=True)
    outputs, h_fw, h_bw = gru(x)

    torch.manual_seed(100304343)
    x = torch.randn(5, 3, 10)
    gru2 = nn.GRU(10, 20, bidirectional=True, batch_first=True)
    outputs, h = gru2(x)
    
    print("Checking the bidirectional GRU implementation")
    print("Same hidden states of the forward cell?\t\t{}".format(
        is_identical(h[0].detach().numpy(), h_fw.detach().numpy())
    ))
    print("Same hidden states of the backward cell?\t{}".format(
        is_identical(h[1].detach().numpy(), h_bw.detach().numpy())
    ))
