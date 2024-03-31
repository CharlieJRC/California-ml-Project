import pandas as pd 

#loading the data
def load_housing_data():
    return pd.read_csv("housing.csv")

housing = load_housing_data()


#using plt to get general understand of what data looks like
#shows that some of the attributes are skewed right
import matplotlib.pyplot as plt
# housing.hist(bins=50, figsize=(10,6))


"""Creating train and test sets. The way the sets are collection is through stratifing based on median income.
This ensures that both sets are represintaive of the whole population"""
#added incomecat to housing wiich is describes the category as declared below
import numpy as np
housing["income_cat"] = pd.cut(housing["median_income"],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])
"""visulizing the categories for understanding (shows how many entries are in each income category"""
# housing["income_cat"].value_counts().sort_index().plot.bar(rot=0)
# plt.xlabel("Income Category")
# plt.ylabel("Number of Houses")
plt.show()


#randomly selects train/test sets that are repesenative of income_cat
#do stratified sampling. Multiple splits used to better estimate model preformance
from sklearn.model_selection import StratifiedShuffleSplit
splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.20, random_state=42)
strat_splits = []
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

strat_train_set, strat_test_set = strat_splits[0]
#print(strat_test_set["income_cat"].value_counts()) #preportional to graph above

