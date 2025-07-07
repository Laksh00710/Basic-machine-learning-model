# Basic-machine-learning-model
pip install scikit-learn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Train model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Test accuracy
score = clf.score(X_test, y_test)
print(f"Model accuracy: {score:.2f}")
