import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Assuming df_clean is already cleaned and loaded properly
# Example structure for df_clean:
# df_clean = pd.DataFrame({
#     'Area': [...],
#     'Bedrooms': [...],
#     'Price': [...]
# })

# Using cleaned data from Part A
X = df_clean[['Area', 'Bedrooms']]
y = df_clean['Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Predict sample
sample = pd.DataFrame({'Area': [1500], 'Bedrooms': [3]})
print("Predicted Price for Area=1500 & Bedrooms=3:", model.predict(sample)[0])




















'''
# A

from scipy.stats import ttest_ind

# Time data
group1 = [85, 95, 100, 80, 90, 97, 104, 95, 88, 92, 94, 99]
group2 = [83, 85, 96, 92, 100, 104, 94, 95, 88, 90, 93, 94]

# Two-sample T-Test (assuming equal variances)
t_stat, p_value = ttest_ind(group1, group2)

print("T-statistic:", t_stat)
print("P-value:", p_value)

# Interpretation
if p_value < 0.05:
    print("Reject the null hypothesis: There is a significant difference.")
else:
    print("Fail to reject the null hypothesis: No significant difference.")



# B

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

# Load Pima Indian Diabetes dataset
pima = fetch_openml(name='diabetes', version=1, as_frame=True)
df = pima.frame

# Features and target
X = df.drop('class', axis=1)  # use all columns except target
y = df['class'].apply(lambda x: 1 if x == 'tested_positive' else 0)  # binary target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and check score
score = model.score(X_test, y_test)
print("Model R² Score:", score)


'''