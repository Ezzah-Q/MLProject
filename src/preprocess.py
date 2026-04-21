from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Imbalance Handling using SMOTE
def apply_smote(X_train, y_train):
    smote = SMOTE(random_state = 42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

# Scale 
def scale_features(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler