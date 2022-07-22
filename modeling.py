import pandas as pd

X = pd.read_csv('final_X.csv')
y = pd.read_csv('final_y.csv')

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=42, test_size=0.20)

import xgboost as xgb

xgb_model = xgb.XGBRegressor(colsample_bytree=0.5,
                              subsample=0.8,
                              gamma = 0.4,
                              max_depth = 3,
                              min_child_weight = 7,
                              reg_lambda=0.8,
                              reg_alpha=54,
                              eta=0.2,
                              n_estimators = 200, verbosity=0)

evaluation = [( X_train, y_train), ( X_valid, y_valid)]

xgb_model.fit(X_train, y_train, eval_set=evaluation, eval_metric="rmse", early_stopping_rounds=10, verbose=False)

import pickle
pick_insert = open('xgbmodel.pkl','wb')
pickle.dump(xgb_model, pick_insert)
pick_insert.close()
    
