import numpy as np

# Define activation functions (sigmoid + tanh)
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
 
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)
 
def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

class LSTMCell:
    """ 
    One LSTM cell that uses 4 gates 

    Gates:
        f - forget gate : what to erase from cell state
        i - input gate: how much new input to store
        g - cell gate: what new input to store 
        o - output gate: what to pass to next step

    Each gate has W (weights) and b (bias)
    """

    # initialization
    def __init__(self, input_size, hidden_size):
        # number of features at each timestep
        self.input_size  = input_size
        # size of hidden state
        self.hidden_size = hidden_size

        # stable initalization
        scale = 1 / np.sqrt(hidden_size)

        # initialize forget gate weight Wf and bias bf
        self.Wf = np.random.uniform(-scale, scale, (hidden_size, hidden_size + input_size))
        self.bf = np.zeros((hidden_size, 1))

        # initialize input gate weight Wi and bias bi
        self.Wi = np.random.uniform(-scale, scale, (hidden_size, hidden_size + input_size))
        self.bi = np.zeros((hidden_size, 1))

        # initialize cell gate weight Wg and bias bg
        self.Wg = np.random.uniform(-scale, scale, (hidden_size, hidden_size + input_size))
        self.bg = np.zeros((hidden_size, 1))

        # initialize output gate weight Wo and bias bo
        self.Wo = np.random.uniform(-scale, scale, (hidden_size, hidden_size + input_size))
        self.bo = np.zeros((hidden_size, 1))
    
    """
    One forward step through the LSTM cell
 
    Inputs:
        x - input at current timestep, shape (input_size, 1)
        h_prev - hidden state from previous step, shape (hidden_size, 1)
        c_prev - cell state from previous step, shape (hidden_size, 1)
 
    Outputs:
        h_next - new hidden state
        c_next - new cell state
        cache - values saved for backpropagation
        """
    def forward(self, x, h_prev, c_prev):
        # concatenate previous hidden state and current input
        concat = np.vstack((h_prev, x))  

        # gate calculations
        f_raw = self.Wf @ concat + self.bf
        i_raw = self.Wi @ concat + self.bi
        g_raw = self.Wg @ concat + self.bg
        o_raw = self.Wo @ concat + self.bo

        # forget gate
        f = sigmoid(f_raw)   
        # input gate        
        i = sigmoid(i_raw)
        # cell gate (candidate values)             
        g = np.tanh(g_raw) 
        # output gate            
        o = sigmoid(o_raw)             
 
        # update cell state (long term memory) and hidden state (short term memory)
        c_next = f * c_prev + i * g   
        h_next = o * np.tanh(c_next)   
 
        # store what we need for backpropagation
        cache = (concat, f, i, g, o, f_raw, i_raw, g_raw, o_raw, c_prev, c_next)
        return h_next, c_next, cache

    """
        Backpropagation through time (BPTT) for one LSTM cell step.
 
        Inputs:
            dh_next - gradient of loss w.r.t. h_next
            dc_next - gradient of loss w.r.t. c_next
            cache - saved values from forward pass
 
        Outputs:
            dx - gradient w.r.t. input x
            dh_prev - gradient w.r.t. h_prev
            dc_prev - gradient w.r.t. c_prev
            grads - dict of gradients for all weights and biases
        """
    def backward(self):
        
        # unpacking the cache
        concat, f, i, g, o, f_raw, i_raw, g_raw, o_raw, c_prev, c_next = cache
 
        # output gate gradient
        do      = dh_next * np.tanh(c_next)
        do_raw  = do * sigmoid_derivative(o_raw)
 
        # cell state gradient
        dc      = dh_next * o * tanh_derivative(c_next) + dc_next
        dc_prev = dc * f
 
        # forget gate gradient
        df      = dc * c_prev
        df_raw  = df * sigmoid_derivative(f_raw)
 
        # input gate gradient
        di      = dc * g
        di_raw  = di * sigmoid_derivative(i_raw)
 
        # cell gate gradient
        dg      = dc * i
        dg_raw  = dg * tanh_derivative(g_raw)
 
        # weight gradients 
        grads = {
            'Wf': df_raw @ concat.T,  'bf': df_raw,
            'Wi': di_raw @ concat.T,  'bi': di_raw,
            'Wg': dg_raw @ concat.T,  'bg': dg_raw,
            'Wo': do_raw @ concat.T,  'bo': do_raw,
        }
 
        # backpropagation into input + hidden
        d_concat = (self.Wf.T @ df_raw +
                    self.Wi.T @ di_raw +
                    self.Wg.T @ dg_raw +
                    self.Wo.T @ do_raw)
 
        dh_prev = d_concat[:self.hidden_size]
        dx      = d_concat[self.hidden_size:]
 
        return dx, dh_prev, dc_prev, grads
    
class LSTMModel:
    """
    Multi-layer LSTM model for binary classification (fraud detection).
 
    Architecture:
        Input → [LSTM layers] → Final hidden state → Linear → Sigmoid → P(fraud)
    """