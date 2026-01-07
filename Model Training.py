"""
Created on 1/7/26
@author: zevvanzanten
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from joblib import dump
import ssl
import os

#Fixing SSL
ssl._create_default_https_context = ssl._create_unverified_context

#Creating models folder
os.makedirs("models", exist_ok=True)

#Loading dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

#Selecting features
X = df[['HouseAge','AveRooms','AveBedrms']]
y = df['MedHouseVal']

#Creating models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=500,max_depth=15,random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=500,learning_rate=0.05,max_depth=5,random_state=42),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'Linear Regression': LinearRegression()
}

#Fitting models
for name, model in models.items():
    model.fit(X,y)
    #Saving model
    dump(model,f"models/{name.replace(' ','_').lower()}.joblib")
    print(f"✅Saving {name} completed")

if __name__ == '__main__':
    pass
