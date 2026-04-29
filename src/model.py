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

            # forward pass through each layer
            for layer in range(self.num_layers):
                if layer == 0:
                    input_to_layer = x_t
                    
            # for the first layer, input is the raw data at this timestep
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
        
        
        #  values for backpropagation later
        cache_out = (caches, h_final, y_raw, X_seq)
        return y_pred, cache_out
   
    # backprop through time across all layers and all timesteps
    def backward(self, y_pred, y_true, cache_out):
        caches, h_final, y_raw, X_seq = cache_out # we saved this from the forward pass so we can compute gradients
        T = X_seq.shape[0] # number of timesteps in this sequence

        # binary cross entropy gradient w.r.t. y_raw simplifies to (y_pred - y_true)
        dy_raw = y_pred - y_true

        # gradients for the dense output layer
        dWy = dy_raw @ h_final.T
        dby = dy_raw
        dh_final = self.Wy.T @ dy_raw

        # accumulate gradients across timesteps for each layer
        # one dict per layer, holding summed grads
        layer_grads = [
            {'Wf': np.zeros_like(cell.Wf), 'bf': np.zeros_like(cell.bf),
             'Wi': np.zeros_like(cell.Wi), 'bi': np.zeros_like(cell.bi),
             'Wg': np.zeros_like(cell.Wg), 'bg': np.zeros_like(cell.bg),
             'Wo': np.zeros_like(cell.Wo), 'bo': np.zeros_like(cell.bo)}
            for cell in self.cells
        ]

        # at the final timestep, only the top layer gets the gradient from the output layer
        # for the rest of the layers and timesteps, the gradient from the output layer is zero
        dh_next = [np.zeros((self.hidden_size, 1)) for _ in range(self.num_layers)]
        dc_next = [np.zeros((self.hidden_size, 1)) for _ in range(self.num_layers)]
        dh_next[-1] = dh_final   # only top layer at last timestep

        # walk backwards through time
        for t in reversed(range(T)):
            layer_caches = caches[t]
            # at each timestep we also walk top-down through the layer stack
            dx_from_above = None
            for layer in reversed(range(self.num_layers)):
                # if not the top layer, the gradient from the layer above (same t) adds in
                if dx_from_above is not None:
                    dh_next[layer] = dh_next[layer] + dx_from_above
                    
                # compute gradients for this layer at this timestep
                dx, dh_prev, dc_prev, grads = self.cells[layer].backward(
                    dh_next[layer], dc_next[layer], layer_caches[layer]
                )

                # accumulate
                for k in layer_grads[layer]:
                    layer_grads[layer][k] += grads[k]

                # set up gradients for the previous timestep
                dh_next[layer] = dh_prev
                dc_next[layer] = dc_prev

                # this dx feeds into the layer below (at the same timestep)
                dx_from_above = dx

        return layer_grads, dWy, dby

    def train(self, X_train_seq, y_train_seq, epochs):
        epoch_losses = []

        # loop through each epoch and restart total_loss
        for epoch in range(epochs):
            total_loss = 0

        # loop through each window sequence for X_train 
            for i in range(len(X_train_seq)):
                # get the sequence window
                X_sequence = X_train_seq[i]
                # get the y output for that sequence
                y_output = y_train_seq[i]

                # run the LSTMmodel forward pass, get back fraud probability
                y_predict, cache_out = self.forward(X_sequence)

                # if y_output = 1 (yes fraud) then y_predict = 0.0 --> make sure log(0) don't occur in entropy loss calculation
                y_predict_clipped = np.clip(y_predict, 1e-7, 1 - 1e-7)
                # compute the binary cross entropy loss (-yi x log(p(yi)) + (1-yi) x log(1-p(yi)))
                loss = -(y_output * np.log(y_predict_clipped) + (1 - y_output) * np.log(1 - y_predict_clipped))
                total_loss += loss.item()

                # start backpropogation --> calculate gradient of the loss
                gradient = (y_predict - y_output) / (y_predict_clipped * (1 - y_predict_clipped))
                # unpack cache from the forward pass
                caches, h_final, y_raw, X_seq_saved = cache_out

                # calculate how much we need to adjust the weights and bias of final layer to reduce loss
                dWy = gradient * sigmoid_derivative(y_raw) * h_final.T
                dby = gradient * sigmoid_derivative(y_raw)
                # final error we send back toward other LSTM layers
                dh_final = self.Wy.T * gradient * sigmoid_derivative(y_raw)

                # update the outer layer weights and bias 
                self.Wy -= self.lr * dWy
                self.by -= self.lr * dby

                # initialize gradients for hidden and cell state to 0
                dh = [np.zeros((self.hidden_size, 1)) for _ in range(self.num_layers)]
                dc = [np.zeros((self.hidden_size, 1)) for _ in range(self.num_layers)]
                dh[-1] = dh_final

                # loop from last timestep to the first one
                for t in reversed(range(len(caches))):
                    layer_caches = caches[t]

                    # loop backwards through the layers
                    for layer in reversed(range(self.num_layers)):
                        # get the gradients for that cell's weights and error to pass back to prev layer
                        dx, dh[layer], dc[layer], grads = self.cells[layer].backward(
                            dh[layer], dc[layer], layer_caches[layer]
                        )

                        # update all the weights and biaes for that LSTM cell
                        for param in ['Wf', 'bf', 'Wi', 'bi', 'Wg', 'bg', 'Wo', 'bo']:
                            updated = getattr(self.cells[layer], param) - self.lr * grads[param]
                            setattr(self.cells[layer], param, updated)

            # for all epochs compute avg loss
            avg_loss = total_loss / len(X_train_seq)
            epoch_losses.append(avg_loss)

        # print table at the end
        print(f"\n{'Epoch':<10} {'Avg Loss':<10}")
        print(f"{'-'*20}")
        for epoch, loss in enumerate(epoch_losses):
            print(f"{epoch+1:<10} {loss:<10.4f}")

    def predict(self, X_test_seq):
        # two empty lists: predict is for fraud/no fraud (1/0), prob is for the probabilites
        y_predict_list = []
        y_prob_list = []

        # loop through every test sequence
        for idx in range(len(X_test_seq)):
            X_sequence = X_test_seq[idx]
            # run the forward pass (we dont need cache)
            y_prob, _ = self.forward(X_sequence)

            # store probability and convert it to either 0 or 1 using 0.5 as cutoff
            y_prob_list.append(y_prob.item())
            y_predict_list.append(1 if y_prob.item() >= 0.5 else 0)

        # convert to numpy arrays
        return np.array(y_predict_list), np.array(y_prob_list)
