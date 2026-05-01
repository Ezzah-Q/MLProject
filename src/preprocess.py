from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Imbalance Handling using SMOTE
def apply_smote(X_train, y_train):
    # Initialize SMOTE to random state
    smote = SMOTE(random_state = 42)

    # Balance
    X_sample, y_sample = smote.fit_resample(X_train, y_train)
    
    # Return 
    return X_sample, y_sample

# Scale 
def scale_features(X_train, X_test):
    # Initialize scaler
    scaler = StandardScaler()

    # Fit training data and transform train and test data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Return scaled train and tested data
    return X_train_scaled, X_test_scaled, scaler