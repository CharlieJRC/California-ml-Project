#This file is used to gain insights about the data via the means of graphs
#This will aid in selecting an apporiate model
import matplotlib.pyplot as plt
from load_data import strat_splits

strat_train_set = strat_splits[0][0]
housing = strat_train_set.copy()


#version 1
housing.plot(kind="scatter",  x="longitude", y="latitude", grid=True)

#version 2 - allows us to see density of points
housing.plot(kind="scatter",  x="longitude", y="latitude", grid=True, alpha=0.2)

#version 3 - allows to see color pased on price of house, radius is the districts population
#shows how house prices are related to income
housing.plot(kind="scatter", x="longitude",  y="latitude", s=housing["population"] / 100,
                label="Population", c="median_house_value", cmap="jet", colorbar=True,
                legend=True, sharex=False, figsize=(10,7))

#how each variable corrilates to median house value
corr_matrix = housing.corr(numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))


#using pandas to plot corrilation between certain variables
#will give insights  into which features may be useful for predicting house prices
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))

#from previous figure its clear that income has a correlation with house value
#this graph points out lines in data at certain pricepoints that i will need to clean up
housing.plot(kind="scatter", x="median_income",  y="median_house_value", grid=True, alpha=0.1)
plt.show()


#adding usefull attriubtes and then checking correlation
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]

#shows that rooms_per_house and bedrooms_ratio have stronger correlations and are more useful to us
corr_matrix = housing.corr(numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))