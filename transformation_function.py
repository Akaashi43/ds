import pandas as pd

# Read CSV
df = pd.read_csv('cars.csv')

# Filter Buy Price >= 3000
filtered = df[df['BuyPrice'] >= 3000]
print("Filtered Records:\n", filtered)

# Sort ascending by BuyPrice
sorted_df = df.sort_values(by='BuyPrice')
print("\nSorted Data:\n", sorted_df)

# Group by Model
grouped = df.groupby('Model').mean(numeric_only=True)
print("\nGrouped by Model:\n", grouped)
