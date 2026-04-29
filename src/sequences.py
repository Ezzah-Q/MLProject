import numpy as np

def create_sequences(X, y, window_size):
    sequences = []
    labels = []

    # make small windows of data so the LSTM can look at past rows
    for i in range(len(X) - window_size):
        seq = X[i : i + window_size]
        label = y[i + window_size]

        sequences.append(seq)
        labels.append(label)

    return np.array(sequences), np.array(labels)