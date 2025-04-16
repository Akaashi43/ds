import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample Data
data = pd.DataFrame({
    'Height': [150, 160, 170, 180, 190],
    'Weight': [45, 55, 65, 75, 85]
})

# Linear Regression
X = data[['Height']]
y = data['Weight']
model = LinearRegression()
model.fit(X, y)

# Prediction
print("Predicted weight for 175 cm:", model.predict([[175]])[0])

# Plot
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Linear Regression: Height vs Weight")
plt.grid()
plt.show()
