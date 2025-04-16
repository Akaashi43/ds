from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

# Load dataset
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)

# Standardization
standard_scaler = StandardScaler()
df_standardized = pd.DataFrame(standard_scaler.fit_transform(df), columns=df.columns)

# Normalization
minmax_scaler = MinMaxScaler()
df_normalized = pd.DataFrame(minmax_scaler.fit_transform(df), columns=df.columns)

# Show samples
print("Standardized Data (first 5 rows):")
print(df_standardized.head())

print("\nNormalized Data (first 5 rows):")
print(df_normalized.head())
