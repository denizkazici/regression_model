import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBClassifier

data = pd.read_csv('regression_model/4.csv')


print(data.head())

print(data.isnull().sum())


x = data.iloc[:, :-1].values
x.shape

y = data.iloc[:, -1].values
y.shape

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

pipeline = Pipeline([
    ('scaler', StandardScaler()),  
    ('regression', LinearRegression())  
])

pipelinexgb = Pipeline([
    ('scaler', StandardScaler()),  
    ('model', xgb.XGBRegressor())  
])
param_grid_xgb = {
    'model__max_depth': [3, 4, 5, 6, 7, 8],
    'model__min_child_weight': [1, 5, 10],
    'model__gamma': [0.5, 1, 1.5, 2, 5],
    'model__colsample_bytree': [0.6, 0.8, 1.0],
    'model__learning_rate': [0.01, 0.1, 0.2, 0.3]
}

grid = GridSearchCV(estimator=pipelinexgb, param_grid=param_grid_xgb, cv=3, scoring='neg_mean_squared_error', n_jobs=1)
grid.fit(x_train, y_train)
bestmodel = grid.best_estimator_
y_predXGB = bestmodel.predict(x_test)


pipeline.fit(x_train, y_train)
y_predLinear = pipeline.predict(x_test)

mse_lr = mean_squared_error(y_test, y_predLinear)
mse_xgb = mean_squared_error(y_test, y_predXGB)

r2_lr = r2_score(y_test, y_predLinear)
r2_xgb = r2_score(y_test, y_predXGB)
print("RESULTS:")
print("Linear Regression MSE:", mse_lr)
print("Linear Regression R^2:", r2_lr)
print("XGBoost MSE:", mse_xgb)
print("XGBoost R^2:", r2_xgb)