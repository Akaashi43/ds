import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
import seaborn as sns

# Load Titanic dataset
titanic = sns.load_dataset("titanic")

# Preprocess
titanic = titanic[['survived', 'pclass', 'sex', 'age']]
titanic.dropna(inplace=True)
titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1})

# Features and target
X = titanic[['pclass', 'sex', 'age']]
y = titanic['survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Decision Tree
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Interpret decision rules
rules = export_text(clf, feature_names=list(X.columns))
print("\nDecision Tree Rules:\n", rules)
