#Aim: Demonstrate the text classifier using Naïve bayes classifier algorithm.
#Program: Write a program to implement the naive Bayesian classifier for a sample training data set stored as a .CSV file. Compute the accuracy of the classifier, considering few test data sets.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

# Load data
df = pd.read_csv("5.csv")

# Define feature columns and predicted class names
feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
predicted_class_names = ['diabetes']

# Prepare features and target variable
X = df[feature_col_names].values
y = df[predicted_class_names].values.ravel()  # Flatten y to a 1D array

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Print number of training and test data
print("Total number of Training Data:", y_train.shape[0])
print("Total number of Test Data:", y_test.shape[0])

# Train Naive Bayes (NB) classifier
clf = GaussianNB()
clf.fit(X_train, y_train)

# Predictions
predicted = clf.predict(X_test)
predict_test_data = clf.predict([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])

# Model Evaluation
print('\nConfusion matrix:')
print(metrics.confusion_matrix(y_test, predicted))
print('\nAccuracy of the classifier:', metrics.accuracy_score(y_test, predicted))
print('\nPrecision:', metrics.precision_score(y_test, predicted))
print('\nRecall:', metrics.recall_score(y_test, predicted))
print("\nPredicted Value for individual Test Data:", predict_test_data)


