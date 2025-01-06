import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Read the dataset
data = pd.read_csv('500_Person_Gender_Height_Weight_Index.csv')
data.head()

# Describe the dataset
data.describe()

# Function to map numeric index to string categories
def give_names_to_indices(ind):
    if ind == 0:
        return 'Extremely Weak'
    elif ind == 1:
        return 'Weak'
    elif ind == 2:
        return 'Normal'
    elif ind == 3:
        return 'OverWeight'
    elif ind == 4:
        return 'Obesity'
    elif ind == 5:
        return 'Extremely Obese'

# Apply the function to the 'Index' column
data['Index'] = data['Index'].apply(give_names_to_indices)

# Plot the relationship between Height and Weight
sns.lmplot('Height', 'Weight', data, hue='Index', height=7, aspect=1, fit_reg=False)

# Count people by gender
people = data['Gender'].value_counts()
print(people)

# Count people by BMI categories
categories = data['Index'].value_counts()
print(categories)

# STATS FOR MEN
stats_men = data[data['Gender'] == 'Male']['Index'].value_counts()
print(stats_men)

# STATS FOR WOMEN
stats_women = data[data['Gender'] == 'Female']['Index'].value_counts()
print(stats_women)

# One-hot encode the Gender column
data2 = pd.get_dummies(data['Gender'])
data.drop('Gender', axis=1, inplace=True)
data = pd.concat([data, data2], axis=1)

# Separate features and target variable
y = data['Index']
data = data.drop(['Index'], axis=1)

# Standardize the data
scaler = StandardScaler()
data = scaler.fit_transform(data)
data = pd.DataFrame(data)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=101)

# Define the model and hyperparameters for GridSearchCV
param_grid = {'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 1000]}
grid_cv = GridSearchCV(RandomForestClassifier(random_state=101), param_grid, verbose=3)

# Train the model
grid_cv.fit(X_train, y_train)

# Print best parameters from GridSearchCV
print("Best Parameters: ", grid_cv.best_params_)

# Make predictions
pred = grid_cv.predict(X_test)

# Evaluate the model
print(classification_report(y_test, pred))
print('\n')
print(confusion_matrix(y_test, pred))
print('\n')
print('Accuracy is --> ', accuracy_score(y_test, pred) * 100)
print('\n')

# Live prediction function
def lp(details):
    gender = details[0]
    height = details[1]
    weight = details[2]

    # One-hot encode gender
    if gender == 'Male':
        details = np.array([[np.float(height), np.float(weight), 0.0, 1.0]])
    elif gender == 'Female':
        details = np.array([[np.float(height), np.float(weight), 1.0, 0.0]])

    # Predict the BMI category
    y_pred = grid_cv.predict(scaler.transform(details))
    return (y_pred[0])

# Live predictor example
your_details = ['Male', 175, 80]  # Input details
print("Predicted BMI Category: ", lp(your_details))
