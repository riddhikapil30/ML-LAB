#Write a program to construct a Bayesian network considering medical data. Use this model to demonstrate the diagnosis of heart patients using standard Heart Disease Data Set. You can use Python ML library classes/API.
import pandas as pd
from sklearn.model_selection import train_test_split
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Load data from CSV file
heart_data = pd.read_csv("6.csv")

# Discretize the 'age' variable into categories
age_bins = [20, 40, 60, 80]
age_labels = ['20-39', '40-59', '60-79']
heart_data['age'] = pd.cut(heart_data['age'], bins=age_bins, labels=age_labels)

# Convert columns to categorical types
for col in heart_data.columns:
    heart_data[col] = heart_data[col].astype('category')

# Display the first few rows of the dataset
print(heart_data.head())

# Split the data into training and testing sets
train_data, test_data = train_test_split(heart_data, test_size=0.2, random_state=42)

# Define the structure of the Bayesian Network
model = BayesianNetwork([('age', 'trestbps'),
                         ('age', 'fbs'),
                         ('sex', 'trestbps'),
                         ('trestbps', 'heart_disease'),
                         ('chol', 'heart_disease'),
                         ('fbs', 'heart_disease')])

# Fit the model using Maximum Likelihood Estimation
model.fit(train_data, estimator=MaximumLikelihoodEstimator)

# Perform inference
infer = VariableElimination(model)

# Query the model to calculate the probability of heart disease given new data
query_result = infer.query(variables=['heart_disease'], evidence={
    'age': '40-59',  # Use discrete age category
    'sex': 1,
    'chol': 250,
    'trestbps': 130,
    'fbs': 0
})

print(query_result)

