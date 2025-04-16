import pandas as pd

# Step 1: Read CSV
df = pd.read_csv("cars.csv")

# Step 2: Filter SellPrice > 4000
filtered = df[df['SellPrice'] > 4000]
print("Cars with Sell Price > 4000:\n", filtered)

# Step 3: Sort in ascending order (by SellPrice)
sorted_df = df.sort_values(by='SellPrice')
print("\nSorted Data:\n", sorted_df)

# Step 4: Group by Make
grouped = df.groupby('Make').mean(numeric_only=True)
print("\nGrouped by Make:\n", grouped)
