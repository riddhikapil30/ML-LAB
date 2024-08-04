#Aim: To construct the Decision tree using the training data sets under supervised learning concept.
#Program: Write a program to demonstrate the working of the decision tree based ID3 algorithm. Use an appropriate data set for building the decision tree and apply this knowledge to classify a new sample.
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the DecisionTreeClassifier with the ID3 criterion (entropy)
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)

# Fit the model on the training data
clf.fit(X_train, y_train)

# Predicting a new sample
# Let's assume a new sample with arbitrary measurements
new_sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = clf.predict(new_sample)
print(f"Predicted class for the new sample: {iris.target_names[prediction[0]]}")

# Plotting the decision tree
plt.figure(figsize=(20, 10))
tree.plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, fontsize=7, proportion=False)
plt.show()
