from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Use petal length and petal width
X = df[['petal length (cm)']]
y = df['petal width (cm)']

# Train model
model = LinearRegression()
model.fit(X, y)

# Prediction example
pred = model.predict([[4.5]])
print(f"Predicted petal width for petal length 4.5 cm: {pred[0]:.2f} cm")

# Visualization
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Linear Regression: Petal Width vs Petal Length")
plt.grid()
plt.show()
