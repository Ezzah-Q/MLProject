import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
 
# Import each team member's file 
#from load_data   import load_data                  
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

    # load features (X) and labels (y) from data
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    # Split into features and label
    X = df.drop(columns=['Class']).values
    y = df['Class'].values

    # split into training and testing data
    print("\nSplit into train/test data")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # preprocess data
    print("\nPreprocessing")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    X_train_balanced, y_train_balanced    = apply_smote(X_train_scaled, y_train)

    # create sequences
    print("\nCreating sequences...")
    X_train_seq, y_train_seq = create_sequences(X_train_balanced, y_train_balanced, WINDOW_SIZE)
    X_test_seq, y_test_seq   = create_sequences(X_test_scaled, y_test, WINDOW_SIZE)
    

    # build and train model
    print("\nTraining model...")
    input_size = X_train_seq.shape[2]
    model = LSTMModel(
    input_size=input_size,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    learning_rate=LEARNING_RATE
)

    input_size = X_train_seq.shape[2]

    model = LSTMModel(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        learning_rate=LEARNING_RATE,
    )

    model.train(X_train_seq, y_train_seq, epochs=EPOCHS)

    # evaluate model
    print("\nEvaluating on test set...")


if __name__ == "__main__":
    main()
    