import pandas as pd
import numpy as np
import streamlit as st


def rf(HP, new_data):
    # Applying the Random Forest Algorithm 
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    # Split data into training and testing sets
    train_data, test_data = train_test_split(HP, test_size=0.2, random_state=42)

    # Define features and target variable
    features = ['SQFT', 'BHK', 'NEW_LOCATION', 'NEW_Type']
    target = 'ACT_PRICE'


    # Create a Random Forest model
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42)

    # Train the model
    rf_model.fit(train_data[features], train_data[target])

    # Make predictions on the test set
    predictions = rf_model.predict(test_data[features])
  
    return rf_model.predict(new_data)
    

def lr(HP,new_data):
	# Applying the Linear Regression Algorithm 
	X = HP[['SQFT','BHK','NEW_LOCATION','NEW_Type']]
	y=HP['ACT_PRICE']
	from sklearn.model_selection import train_test_split 
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
	from sklearn.linear_model import LinearRegression
	lm = LinearRegression()	
	lm.fit(X_train,y_train)
	predictions = lm.predict(X_test)
	
	return lm.predict(new_data)
	
def svma(HP,new_data):
	# Applying the SVM Algorithm
	from sklearn import svm
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import mean_squared_error
	X = HP[['SQFT','BHK','NEW_LOCATION','NEW_Type']]
	y=HP['ACT_PRICE']
	# Split the data into training and testing datasets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	clf = svm.SVR(kernel='linear', C=1.0, epsilon=0.1)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	
	return clf.predict(new_data)

def dest(HP,new_data):
	# Applying Decision tree Algoritm 
	from sklearn.tree import DecisionTreeRegressor
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import mean_squared_error
	X = HP[['SQFT','BHK','NEW_LOCATION','NEW_Type']]
	y=HP['ACT_PRICE']
	# Split the data into training and testing datasets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Train the Decision Tree model
	dt = DecisionTreeRegressor(max_depth=3, random_state=42)
	dt.fit(X_train, y_train)
	# Evaluate the model
	y_pred = dt.predict(X_test)
	
	return dt.predict(new_data)
