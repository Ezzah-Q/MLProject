import pandas as pd

df = pd.read_csv("data/creditcard.csv")

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