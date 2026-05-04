import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import gdown
from pathlib import Path

# Import each team member's file               
from preprocess  import scale_features, apply_smote  
from sequences   import create_sequences            
from model       import LSTMModel                  
from evaluation  import evaluate_model         

# seed for reproducibility (so weight init + SMOTE stay consistent across runs)
np.random.seed(42)

# hyperparameters (tune these)
WINDOW_SIZE = 5        # how many transactions the model sees at once
HIDDEN_SIZE = 64       # size of the hidden state vector
NUM_LAYERS = 1         # number of stacked LSTM layers
LEARNING_RATE = 0.0001  # how fast the model updates weights
EPOCHS = 10            # how many times to loop through training data


def main():
    # if the user does not type in a dataset path, download the dataset from google drive
    if len(sys.argv) == 1:
        print("downloading dataset from google drive...")
        GOOGLE_DRIVE_FILE_ID = "1-BQCqkGzJLl_ARyi3Q1U1mc03aXc6DNs"
        data_path = "creditcard.csv"
        gdown.download(id=GOOGLE_DRIVE_FILE_ID, output=data_path, quiet=False)
    # if the user types in a dataset path, use that instead
    elif len(sys.argv) == 2:
        # get the dataset path from the second command line if it is provided
        data_path = sys.argv[1]
    # else, if the user types in more than 2 command line arguments, print usage and exit
    else:
        print("Usage: python main.py <path_to_dataset> (to use existing dataset) || python main.py (to download dataset from google drive)")
        sys.exit(1)

    # load features (X) and labels (y) from data
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
 
    # use stratified sampling to sample randomly from dataset, but keep proportion of fraud to not fraud the same (0.05 equates to about 14,000 rows)
    fraud = df[df['Class'] == 1].sample(frac=0.05, random_state=42)
    normal = df[df['Class'] == 0].sample(frac=0.05, random_state=42)
    df = pd.concat([fraud, normal]).reset_index(drop=True)

    # Split into features and label
    X = df.drop(columns=['Class']).values
    y = df['Class'].values

    # Dataset metrics
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {y.sum()}")

    # split into training and testing data
    print("\nSplit into train/test data")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Training dataset metrics
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    # Preprocess data 
    print("\nPreprocessing")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    X_train_balanced, y_train_balanced    = apply_smote(X_train_scaled, y_train)
    
    # After SMOTE dataset metrics
    print(f"After SMOTE - X_train shape: {X_train_balanced.shape}")
    print(f"After SMOTE - fraud cases: {y_train_balanced.sum()}")

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

    model.train(X_train_seq, y_train_seq, epochs=EPOCHS)

    # evaluate model
    print("\nEvaluating on test set...")
    y_pred, y_prob = model.predict(X_test_seq)
    evaluate_model(y_test_seq, y_pred, y_prob)
    
    # deleting the downloaded dataset
    if len(sys.argv) != 2:
        Path(data_path).unlink()
        print(f"\nDeleted downloaded dataset: {data_path}")

if __name__ == "__main__":
    main()
    
