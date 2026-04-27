import pandas as pd

def load_data(path):
    df = pd.read_csv(path)

    # Basic overview
    print("First 5 rows:")
    print(df.head())

    print("\nShape of dataset:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns)

    print("\nDataset info:")
    print(df.info())

    # Check missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # Class distribution
    print("\nClass distribution:")
    print(df['Class'].value_counts())

    # Percentage distribution
    print("\nClass distribution (%):")
    print(df['Class'].value_counts(normalize=True))

    # Split into features and label
    X = df.drop(columns=['Class']).values
    y = df['Class'].values

    return X, y