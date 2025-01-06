import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
data = pd.read_csv('500_Person_Gender_Height_Weight_Index.csv')

# Map numeric index to string categories
def give_names_to_indices(ind):
    categories = {
        0: 'Extremely Weak',
        1: 'Weak',
        2: 'Normal',
        3: 'Overweight',
        4: 'Obesity',
        5: 'Extremely Obese'
    }
    return categories.get(ind, 'Unknown')

data['Index'] = data['Index'].apply(give_names_to_indices)

# Plot the relationship between Height and Weight
sns.lmplot(x='Height', y='Weight', data=data, hue='Index', height=7, aspect=1, fit_reg=False)
plt.title("Height vs Weight by BMI Category")
plt.show()

# Count people by gender and categories
print("Count by Gender:")
print(data['Gender'].value_counts())

print("\nCount by BMI Categories:")
print(data['Index'].value_counts())

# Stats for men and women
print("\nStats for Men:")
print(data[data['Gender'] == 'Male']['Index'].value_counts())

print("\nStats for Women:")
print(data[data['Gender'] == 'Female']['Index'].value_counts())

# One-hot encode the Gender column
data_gender = pd.get_dummies(data['Gender'], prefix='Gender')
data = pd.concat([data.drop('Gender', axis=1), data_gender], axis=1)

# Separate features and target variable
y = data['Index']
X = data.drop('Index', axis=1)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=101)

# Define the RandomForestClassifier and perform hyperparameter tuning
param_grid = {'n_estimators': [100, 200, 300, 400, 500]}
grid_cv = GridSearchCV(RandomForestClassifier(random_state=101), param_grid, verbose=3)
grid_cv.fit(X_train, y_train)

# Best parameters and model evaluation
print("\nBest Parameters: ", grid_cv.best_params_)

# Make predictions and evaluate
pred = grid_cv.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))
print("\nAccuracy:", accuracy_score(y_test, pred) * 100)

# Live prediction function
def lp(details):
    gender, height, weight = details

    # One-hot encode gender
    if gender == 'Male':
        details = np.array([[float(height), float(weight), 0.0, 1.0]])
    elif gender == 'Female':
        details = np.array([[float(height), float(weight), 1.0, 0.0]])
    else:
        raise ValueError("Gender must be 'Male' or 'Female'.")

    # Standardize input data
    details = scaler.transform(details)

    # Predict BMI category
    y_pred = grid_cv.predict(details)
    return y_pred[0]

# Example live prediction
your_details = ['Male', 175, 80]  # Replace with actual inputs
try:
    predicted_category = lp(your_details)
    print("\nPredicted BMI Category: ", predicted_category)
except Exception as e:
    print(f"Error: {e}")
