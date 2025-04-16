import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample data
data = {'Experience': [1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.2, 3.7],
        'Salary': [39343, 46205, 37731, 43525, 39891, 56642, 60150, 54445, 64445, 57189]}
df = pd.DataFrame(data)

# Linear Regression
X = df[['Experience']]
y = df['Salary']
model = LinearRegression()
model.fit(X, y)

# Prediction
experience = [[4.0]]
predicted_salary = model.predict(experience)
print("Predicted Salary for 4 years experience:", predicted_salary)

# Visualization
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression: Salary vs Experience")
plt.show()
