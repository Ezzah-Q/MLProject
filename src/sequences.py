import numpy as np

# looks at a sliding window of 5 transactions at a time. 
# need to turn the flat array of transactions into overlapping windows that the LSTM can process.

# Instead of feeding the LSTM one transaction at a time, we group
# transactions into overlapping windows of size `window_size`.

"""
  Window 1 -> [A, B, C]  label = C's label
  Window 2 -> [B, C, D]  label = D's label
  Window 3 -> [C, D, E]  label = E's label

The LSTM sees the history of recent transactions and predicts
whether the LAST transaction in the window is fraud or not.

Input shape per sequence: (window_size, num_features)
  - window_size = how many transactions in each window (5 in main.py)
  - num_features = number of columns in X (30 for creditcard.csv)
"""


def create_sequences(X, y, window_size):
    """
    Converts flat transaction arrays into overlapping sliding windows.

    Parameters:
        X          : np.ndarray of shape (num_transactions, num_features)
                     The scaled feature matrix from preprocess.py
        y          : np.ndarray of shape (num_transactions,)
                     The fraud labels (0 = normal, 1 = fraud)
        window_size: int
                     How many transactions per sequence (WINDOW_SIZE in main.py)

    Returns:
        X_seq      : np.ndarray of shape (num_sequences, window_size, num_features)
                     3D array — one window per sequence
        y_seq      : np.ndarray of shape (num_sequences,)
                     Label for each window (label of the LAST transaction in that window)
    """

    X_seq = []
    y_seq = []

    # Slide a window across all transactions
    # Start at index 0, end so last window finishes at the last transaction
    num_transactions = len(X)

    for i in range(num_transactions - window_size + 1):
        # Grab window_size consecutive transactions
        window = X[i : i + window_size]  # shape: (window_size, num_features)

        # The label is for the LAST transaction in the window
        # given these 5 transactions, is the last one fraud?
        label = y[i + window_size - 1]

        X_seq.append(window)
        y_seq.append(label)

    # Convert lists to numpy arrays
    X_seq = np.array(X_seq)  # shape: (num_sequences, window_size, num_features)
    y_seq = np.array(y_seq)  # shape: (num_sequences,)

    # Print summary
    print(f"\nSequence creation complete:")
    print(f"  Input transactions : {num_transactions}")
    print(f"  Window size        : {window_size}")
    print(f"  Total sequences    : {len(X_seq)}")
    print(f"  X_seq shape        : {X_seq.shape}  (sequences, window_size, features)")
    print(f"  y_seq shape        : {y_seq.shape}")
    print(f"  Fraud sequences    : {y_seq.sum()} ({100 * y_seq.mean():.2f}%)")
    print(f"  Normal sequences   : {(y_seq == 0).sum()}")

    return X_seq, y_seq