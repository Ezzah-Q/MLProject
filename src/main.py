import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
 
# Import each team member's file 
from load_data   import load_data                  
from preprocess  import scale_features, apply_smote  
from sequences   import create_sequences            
from model       import LSTMModel                  
from evaluation  import evaluate_model              

# hyperparameters (tune these)
WINDOW_SIZE = 5        # how many transactions the model sees at once
HIDDEN_SIZE = 64       # size of the hidden state vector
NUM_LAYERS = 2         # number of stacked LSTM layers
LEARNING_RATE = 0.001  # how fast the model updates weights
EPOCHS = 10            # how many times to loop through training data

# load data

# preprocess

# create sequences

# model

# evaluate model