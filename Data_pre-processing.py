import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("house_data.csv")

print("Original Data:\n", df)

# Handling missing values
df['Area'].fillna(df['Area'].mean(), inplace=True)
df['Bedrooms'].fillna(df['Bedrooms'].mode()[0], inplace=True)
df['Price'].fillna(df['Price'].median(), inplace=True)

print("\nAfter Handling Missing Values:\n", df)

# Outlier detection using IQR
Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Remove outliers
df_clean = df[(df['Price'] >= lower) & (df['Price'] <= upper)]

print("\nAfter Removing Outliers:\n", df_clean)

# Optional: Visualize outliers
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['Price'])
plt.title("Outliers in House Prices")
plt.show()
