from clean_data import housing, housing_labels, preprocessing
import pandas as pd

#linear model
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

#pipeline for the whole model (preprocessing a pipleine made in clean_data.py)
lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)

housing_predictions = lin_reg.predict(housing)
print(housing_predictions[:5].round(2)) #frist 5 predictions rounded to 2 decimal places
print(housing_labels[:5].values) #first 5 actual values that we are tying to predict

#check the error
from sklearn.metrics import mean_squared_error
lin_rmse = mean_squared_error(housing_labels, housing_predictions, squared=False)
print("Linear Regression RMSE: ", lin_rmse) #model off on averay by roughly 68k

#this model is underfitting the data
#we can try adding more features of a more complex model

#decision tree model
#this model is capable of finding complex non-linear relationships in the data
#it will automatically find and use these features
from sklearn.tree import DecisionTreeRegressor
tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)

tree_housing_prediction = tree_reg.predict(housing)
tree_rmse = mean_squared_error(housing_labels, tree_housing_prediction, squared=False)
print("Decision Tree RMSE: ", tree_rmse) #model is overfitting the data

#cross validation - split the data into multiple parts and train the model on each part
#then average the results to get an overall error rate
from sklearn.model_selection import cross_val_score

tree_rmses = -cross_val_score(tree_reg, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(tree_rmses).describe()) #mean error is 66k with std of 2.1k - this model is not much better than the linear model


#RandomForestRegressor model
#training many decision trees on random subsets of features of the data and averaging the results
#can provide better accuracy than single decision tree
#this is a ensemble - a model made up of many underlying models (in this case decision trees)

from sklearn.ensemble import RandomForestRegressor
forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
forest_rsmes = -cross_val_score(forest_reg, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(forest_rsmes).describe()) # this model is better than the other, but is still overfitting the data


#the idea is to try many models and find handful that are the best
#then fine tune the hyperparameters of those models (hyperparameters are the settings of the model and are not learned during training)

#gridsearchCV good when there arent a tone options for hyperparameters
#but if you have lots of options it can take a long time to run

#randomizedsearchCV is a better option when there are many hyperparameters