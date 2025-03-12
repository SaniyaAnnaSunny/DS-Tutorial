# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
# You can download the dataset from Kaggle or use the one provided by sklearn
# For this example, I'll assume the dataset is in a CSV file named 'diabetes.csv'
data = pd.read_csv('diabetes.csv')

# Display the first few rows of the dataset
print("Dataset Head:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Separate features (X) and target variable (y)
X = data.drop('Outcome', axis=1)  # Features (all columns except 'Outcome')
y = data['Outcome']  # Target variable ('Outcome')

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (mean = 0, variance = 1)
# Logistic regression performs better with standardized data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Predict for unknown data points
# Example: Create a new data point with unknown outcome
unknown_data = pd.DataFrame({
    'Pregnancies': [2],
    'Glucose': [120],
    'BloodPressure': [70],
    'SkinThickness': [30],
    'Insulin': [80],
    'BMI': [25],
    'DiabetesPedigreeFunction': [0.5],
    'Age': [35]
})

# Standardize the unknown data using the same scaler
unknown_data_scaled = scaler.transform(unknown_data)

# Predict the outcome
predicted_outcome = model.predict(unknown_data_scaled)
print("\nPredicted Outcome for Unknown Data:")
print("Diabetes" if predicted_outcome[0] == 1 else "No Diabetes")