import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
 
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


def main():
    # make sure user types in 2 commands
    if len(sys.argv) !=2:
        print("Usage: python main.py <path_to_dataset>")
        print("Example: python main/py data/creditcard.csv")
        sys.exit(1)
    
    # get the dataset path from the second command line
    data_path = sys.argv[1]

    # load data
    print(f"\n Loading data from: {data_path}")

    # split into training and testing data


    # preprocess
    print()

# create sequences

# model

# evaluate model