import numpy as np

# sigmoid activation function used by the gates to decide hou much to let in (squashes everything between 0-1)
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

# derivitives of sigmoid and tanh are used in backpropogation to compute how much to adjust weights 
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

""" 
One LSTM cell that uses 4 gates 
Gates:
    f - forget gate : what to erase from cell state (long term memory)
    i - input gate: how much new input to store
    g - cell gate: what new input to store 
    o - output gate: what to pass to next step
Each gate has W (weights) and b (bias)
W has shape (hidden_size, hidden_size + input_size), in our case that is a matrix of [64, 30]
"""
class LSTMCell:

    # initialization
    def __init__(self, input_size, hidden_size):
        # number of features at each timestep
        self.input_size  = input_size
        
        # size of hidden state
        self.hidden_size = hidden_size

        # keep initial weights small and stable 
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

        # gate calculations: multiple combined input with weight matrix and add bias
        f_raw = self.Wf @ concat + self.bf
        i_raw = self.Wi @ concat + self.bi
        g_raw = self.Wg @ concat + self.bg
        o_raw = self.Wo @ concat + self.bo

        # forget gate: 0=forget, 1=keep
        f = sigmoid(f_raw)   
        # input gate: 0=ignore, 1=store        
        i = sigmoid(i_raw)
        # cell gate (candidate values from -1 to 1)             
        g = np.tanh(g_raw) 
        # output gate            
        o = sigmoid(o_raw)             
 
        # update outputs: cell state (long term memory) and hidden state (short term memory)
        c_next = f * c_prev + i * g   
        h_next = o * np.tanh(c_next)   
 
        # store what we need for backpropagation
        cache = (concat, f, i, g, o, f_raw, i_raw, g_raw, o_raw, c_prev, c_next)
        return h_next, c_next, cache


    """
    Backpropagation through time (BPTT) for one LSTM cell step.
    Inputs:
        dh_next - error from h_next
        dc_next - error from c_next
        cache - saved values from forward pass
    Outputs:
        dx - gradient w.r.t. input x
        dh_prev - gradient w.r.t. h_prev
        dc_prev - gradient w.r.t. c_prev
        grads - dict of gradients for all weights and biases
    """
    def backward(self, dh_next, dc_next, cache):
        # unpack the cache
        concat, f, i, g, o, f_raw, i_raw, g_raw, o_raw, c_prev, c_next = cache
 
        # how much output gate affected loss
        do = dh_next * np.tanh(c_next)
        do_raw = do * sigmoid_derivative(o_raw)
 
        # how much cell state affected loss
        dc = dh_next * o * tanh_derivative(c_next) + dc_next
        dc_prev = dc * f
 
        # how much forget gate affected loss
        df = dc * c_prev
        df_raw = df * sigmoid_derivative(f_raw)
 
        # how much input gate affected loss
        di = dc * g
        di_raw = di * sigmoid_derivative(i_raw)
 
        # how much cell gate affected loss
        dg = dc * i
        dg_raw = dg * tanh_derivative(g_raw)
 
        # dictionary: how much each weight and bias needs to change
        grads = {
            'Wf': df_raw @ concat.T,  'bf': df_raw,
            'Wi': di_raw @ concat.T,  'bi': di_raw,
            'Wg': dg_raw @ concat.T,  'bg': dg_raw,
            'Wo': do_raw @ concat.T,  'bo': do_raw,
        }
 
        # combine gradients from all 4 gates
        d_concat = (self.Wf.T @ df_raw +
                    self.Wi.T @ di_raw +
                    self.Wg.T @ dg_raw +
                    self.Wo.T @ do_raw)
 
        # split combines gradients into dh_prev and dx
        dh_prev = d_concat[:self.hidden_size]
        dx      = d_concat[self.hidden_size:]
 
        return dx, dh_prev, dc_prev, grads
    
class LSTMModel:
    """
    Multi-layer LSTM model for binary classification (fraud detection).
 
    Architecture:
        Input → [LSTM layers] → Final hidden state → Linear → Sigmoid → P(fraud)
    """
    def __init__(self, input_size, hidden_size, num_layers, learning_rate):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lr          = learning_rate

        # build a stack of LSTM cells, one per layer
        # first layer takes the raw input, the rest take the hidden state from below
        self.cells = []
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size
            self.cells.append(LSTMCell(in_size, hidden_size))

        # final dense layer: takes last hidden state and squashes to a single fraud probability
        scale = 1 / np.sqrt(hidden_size)
        self.Wy = np.random.uniform(-scale, scale, (1, hidden_size))
        self.by = np.zeros((1, 1))   
    
    # forward pass through the entire sequence and all layers
    def forward(self, X_seq):
        T = X_seq.shape[0]   # number of timesteps in this sequence

        # initialize hidden and cell states for every layer to zero
        h = []
        c = []
        for i in range(self.num_layers):
            hidden = np.zeros((self.hidden_size, 1))
            cell = np.zeros((self.hidden_size, 1))
            h.append(hidden)
            c.append(cell)
        # we save everything per-timestep, per-layer so we can backprop later
        caches = []

        for t in range(T):
            x_t = X_seq[t].reshape(-1, 1)   # column vector
            layer_caches = []

            for layer in range(self.num_layers):
                if layer == 0:
                    input_to_layer = x_t
                else:
                    input_to_layer = h[layer - 1]
                h[layer], c[layer], cache = self.cells[layer].forward(
                    input_to_layer, h[layer], c[layer]
                )
                layer_caches.append(cache)

            caches.append(layer_caches)

        # final output is based on the last hidden state of the top layer
        h_final = h[-1]
        y_raw = self.Wy @ h_final + self.by
        y_pred = sigmoid(y_raw)
        
        
        #this
        cache_out = (caches, h_final, y_raw, X_seq)
        return y_pred, cache_out

