# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import statsmodels.api as sm  # For statistical modeling, including linear regression

# Load the advertising dataset from the CSV file
# Ensure the file 'Advertising.csv' is in the same directory as this script
data = pd.read_csv('Advertising.csv')

# Define the independent variables (TV, Radio, Newspaper) and the dependent variable (Sales)
# X contains the features (advertising budgets for TV, Radio, and Newspaper)
X = data[['TV', 'radio', 'newspaper']]
# y contains the target variable (sales)
y = data['sales']

# Add a constant term to the independent variables
# This is necessary for the regression model to include an intercept (beta_0)
X = sm.add_constant(X)

# Fit the Ordinary Least Squares (OLS) linear regression model
# OLS is a method for estimating the parameters in a linear regression model
model = sm.OLS(y, X).fit()

# Get the summary of the regression model
# The summary provides detailed statistics about the model's performance
summary = model.summary()

# Extract specific statistics from the model for further analysis
# Residual Standard Error (RSE): Measures the average distance that the observed values fall from the regression line
residual_std_error = model.scale**0.5  # RSE is the square root of the scale (variance of residuals)
# R-squared (R²): Represents the proportion of variance in the dependent variable explained by the independent variables
r_squared = model.rsquared
# F-statistic: Tests the overall significance of the model
f_statistic = model.fvalue

# Print the extracted statistics
print(f"Residual Standard Error: {residual_std_error}")
print(f"R-squared: {r_squared}")
print(f"F-statistic: {f_statistic}")

# Print the full model summary for a comprehensive overview
# This includes coefficients, p-values, confidence intervals, and other diagnostics
print("\nModel Summary:")
print(summary)