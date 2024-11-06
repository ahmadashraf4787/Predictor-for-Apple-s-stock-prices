#!/usr/bin/env python
# coding: utf-8

# <h1 style="color:#008000;font-size:30px">Predictor for Apple's stock prices:</h1>

# <h3 style="color:#8B0000;">Imported All the necessary libraries:</h3>

# In[1]:


import pandas as pd
import numpy as np
#!pip install yfinance
import yfinance as yf
#!pip install pandas_datareader
from pandas_datareader import data as pdr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt


# <h3 style="color:#8B0000;">Define the symbols for the stocks and indices, and Define the start and end dates for the data:</h3>

# In[2]:


# Define the symbols for the stocks and indices to be used in the analysis
symbols = ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', '^GSPC', '^DJI']

# Define the start and end dates for the data
start_date = '2015-01-01'
end_date = '2023-03-31'

# The 'symbols' list contains the tickers of the stocks and indices to be used in the analysis.
# In this case, it includes Apple, Microsoft, Alphabet (Google), Meta Platforms, Amazon, S&P 500 index, and Dow Jones Industrial Average index.

# The 'start_date' and 'end_date' variables determine the time period for which the data will be retrieved.
# In this case, the data will be retrieved from January 1st, 2015 to March 31st, 2023.


# <h3 style="color:#8B0000;">Download the stock and index data from Yahoo Finance:</h3>

# In[3]:


yf.pdr_override()  # Activate yahoo finance workaround
#data = pdr.get_data_yahoo(symbols, start=start_date, end=end_date, tz='UTC')['Adj Close']

data = pdr.get_data_yahoo(symbols, start=start_date, end=end_date, ignore_tz = True)['Adj Close']
data


# <h3 style="color:#8B0000;">Define the target variable (AAPL) and the features (other stocks and indices):</h3>

# In[4]:


# Rename the columns to more meaningful names
data.columns = ['AAPL', 'MSFT', 'GOOGL', 'FB', 'AMZN', 'S&P 500', 'Dow Jones']

# Calculate the percentage change in each variable
pct_change = data.pct_change().dropna()

# Define the target variable (AAPL) and the features (other stocks and indices)
target = 'AAPL'
features = [col for col in pct_change.columns if col != target]


# <h3 style="color:#8B0000;">Define the time delay or lag:</h3>

# In[5]:


# Define the time delay or lag
delay = 1


# <h3 style="color:#8B0000;">Create a lagged dataset by shifting the target variable up by the delay:</h3>

# In[6]:


dataset = pd.concat([pct_change[[target] + features].shift(-delay),
                     pct_change[target]], axis=1)
dataset

#pct_change[[target] + features]: selects the columns in the pct_change dataframe that correspond to the target variable and the features, and calculates the percentage change of these columns.

#.shift(-delay): shifts the selected columns by the value of the delay parameter. This is done so that each row in the resulting dataset corresponds to a specific point in time, and the values of the target and features are shifted forward by the delay value.

#pct_change[target]: selects the column in the pct_change dataframe that corresponds to the target variable, and calculates the percentage change of this column.

#pd.concat([<shifted target and features>, <shifted target>], axis=1): concatenates the shifted target and features, along with the shifted target itself, along the columns axis (axis=1) to form the final dataset.


# <h3 style="color:#8B0000;"> Drop rows with NaN values:</h3>

# In[7]:


dataset.dropna(inplace=True)
#drop the missing values
dataset


# <h3 style="color:#8B0000;"> Split the dataset into training and testing sets:</h3>

# In[8]:


# Features
X = dataset.drop(target, axis=1)
#Response
y = dataset[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# <h3 style="color:#8B0000;">Get the 1-dimentional values for y train and y test sets:</h3>

# In[9]:


# 1-d for y_train
y_train = y_train.iloc[:, 0]
# 1-d for y_test
y_test = y_test.iloc[:, 0]


# <h3 style="color:#8B0000;">Define a custom implementation of multivariate linear regression (with gradient descent): </h3>

# In[10]:


class MultivariateLinearRegression:
    def __init__(self, learning_rate=0.1, max_iterations=1000, tolerance=0.0001):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.theta = None
        self.intercept = None
        
    def fit(self, X, y):
        # Add the intercept term to X
        ones_column = np.ones((X.shape[0], 1))
        X = np.hstack((ones_column, X))
        # Initialize theta and the intercept
        self.theta = np.zeros(X.shape[1])
        self.intercept = self.theta[0]
        # Perform gradient descent
        for iteration in range(self.max_iterations):
            hypothesis = np.dot(X, self.theta)
            error = hypothesis - y
            gradient = np.dot(X.T, error) / X.shape[0]
            self.theta -= self.learning_rate * gradient
            self.intercept = self.theta[0]
            if np.max(np.abs(gradient)) < self.tolerance:
                break
                
    def predict(self, X):
        # Add the intercept term to X
        ones_column = np.ones((X.shape[0], 1))
        X = np.hstack((ones_column, X))
        # Make predictions using dot product
        theta_with_intercept = np.hstack(([self.intercept], self.theta[1:]))
        return np.dot(X, theta_with_intercept)


# <h3 style="color:#8B0000;"> Define the list of models including the custom defined function: </h3>

# In[11]:


models = [
    LinearRegression(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    KNeighborsRegressor(n_neighbors = 50),
    SVR(kernel='linear'),
    MultivariateLinearRegression()
]


# <h3 style="color:#8B0000;"> Train each model and evaluated  using mse evaluation metric:</h3>

# In[12]:


# Define dictionary to store the results of each model
results = {}
r2_lst = []
# Loop through each model, fit the training data, and predict on the test data
for model in models:
    # Get the name of the model class
    model_name = model.__class__.__name__
    # Fit the model to the training data
    model.fit(X_train, y_train)
    # Use the model to predict on the test data
    y_pred = model.predict(X_test)
    
    # Calculate the performance metrics and store them in the results dictionary
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    r2_lst.append(r2)
    results[model_name] = {'mse': mse, 'mae': mae, 'r2': r2}


# <h3 style="color:#8B0000;">  Print the results of each model:</h3>

# In[13]:


# Loop through each model in the dictionary and print out the performance metrics
for model, result in results.items():
    # Print the model name
    print(model + ':')
    # Print the mean squared error
    print('Mean Squared Error: {:.4f}'.format(result['mse']))
    # Print the mean absolute error
    print('Mean Absolute Error: {:.4f}'.format(result['mae']))
    # Print the R-squared score
    print('R^2 Score: {:.4f}'.format(result['r2']))
    # Print a newline character to separate each model's results
    print('\n')


# <h3 style="color:#8B0000;"> Cross validation for Linear Regression Model:</h3>

# In[14]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr_cv_scores = cross_val_score(lr, X_train, y_train, cv=5, scoring = 'r2')

print("Cross-validation scores for Linear Regression:", lr_cv_scores)
print("Mean cross-validation score for Linear Regression:", lr_cv_scores.mean())


# <h3 style="color:#8B0000;"> Cross validation for Decision Tree Regressor Model:</h3>

# In[15]:


from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()
dt_cv_scores = cross_val_score(dt, X_train, y_train, cv=5, scoring = 'r2')

print("Cross-validation scores for Decision Tree Regressor:", dt_cv_scores)
print("Mean cross-validation score for Decision Tree Regressor:", dt_cv_scores.mean())


# <h3 style="color:#8B0000;"> Cross validation for Random Forest Regressor Model:</h3>

# In[16]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf_cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring = 'r2')

print("Cross-validation scores for Random Forest Regressor:", rf_cv_scores)
print("Mean cross-validation score for Random Forest Regressor:", rf_cv_scores.mean())


# <h3 style="color:#8B0000;"> Cross validation for Gradient Boosting Regressor Model:</h3>

# In[17]:


from sklearn.ensemble import GradientBoostingRegressor

gb = GradientBoostingRegressor()
gb_cv_scores = cross_val_score(gb, X_train, y_train, cv=5, scoring = 'r2')

print("Cross-validation scores for Gradient Boosting Regressor:", gb_cv_scores)
print("Mean cross-validation score for Gradient Boosting Regressor:", gb_cv_scores.mean())


# <h3 style="color:#8B0000;"> Cross validation for K-Neighbors Regressor Model:</h3>

# In[18]:


from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor()
knn_cv_scores = cross_val_score(knn, X_train, y_train, cv=5, scoring = 'r2')

print("Cross-validation scores for K-Neighbors Regressor:", knn_cv_scores)
print("Mean cross-validation score for K-Neighbors Regressor:", knn_cv_scores.mean())


# <h3 style="color:#8B0000;"> Cross validation for Support Vector Machine Model:</h3>

# In[19]:


from sklearn.svm import SVR

svr = SVR(kernel = 'linear')
svr_cv_scores = cross_val_score(svr, X_train, y_train, cv=5, scoring = 'r2')

print("Cross-validation scores for Support Vector Regressor:", svr_cv_scores)
print("Mean cross-validation score for Support Vector Regressor:", svr_cv_scores.mean())


# ## Summary:
# This is a Python code for a predictive model that uses various machine learning algorithms to predict the stock prices of Apple. The model uses historical data from Yahoo Finance for several stocks and indices, including Apple, Microsoft, Alphabet (Google), Meta Platforms, Amazon, S&P 500 index, and Dow Jones Industrial Average index. The dataset is preprocessed by calculating the percentage change in each variable and creating a lagged dataset by shifting the target variable up by a given delay. The dataset is then split into training and testing sets, and various regression algorithms are used to predict the stock prices of Apple. The algorithms used include multivariate linear regression, decision tree regression, random forest regression, gradient boosting regression, K-nearest neighbors regression, and support vector regression. The performance of each algorithm is evaluated using metrics such as mean squared error, mean absolute error, and R-squared score.

# In[ ]:




