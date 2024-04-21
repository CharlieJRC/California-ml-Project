from load_data import strat_splits
import numpy as np
import pandas as pd

strat_train_set = strat_splits[0][0]

#want to seperate output variable from inputs when transforming/cleaning data
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


#getting rid of any missing values in the dataset by inserting median value into
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
#all the number attributes in the data set (imputer can only do numbers)
housing_num = housing.select_dtypes(include=[np.number])
#train the imputer on the housing data
imputer.fit(housing_num)
#replace all the missing values with the median // returns numpy array not a dataframe
X = imputer.transform(housing_num)
#add back the labels and reformat it into dataframe // now we have dataframe with all missing values filled in w/ median
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

#now going to deal with the categorical data, in this case "ocean_proximity" 
housing_cat = housing[["ocean_proximity"]]
#convert categories into numerical representations
#using one hot encoder bc differeces in cateogoy dont have correlation (like average, good, better, best)
from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder()
housing_cat_encoded = one_hot_encoder.fit_transform(housing_cat)


#feature scaling/standardization
#some  features are on very different scales so we need to scale them down so that theyre more comparable
#without features with smaller numbers will be ignored
#both scaling and standarization work best when data is realitevly normal. Need to make data nomrally distrubted for scalars to work properly

#if having trouble deciding what type of scaling to use, try both and check the error for both of them

#Min max scaling 
#turns all data into scale from -1 to 1, outliers still have strong effect (scale)
from  sklearn.preprocessing import MinMaxScaler
minMaxScaler = MinMaxScaler(feature_range=(-1,1))
minMaxScaled_data = minMaxScaler.fit_transform(housing_num)

#Standard scaling
#usefuel when data is normally distributed (not scaled)
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
standardScaled_data = standardScaler.fit_transform(housing_num)

#When a feature has a heavy tail/skewed left or right/ doesnt look normal minmax and standard wont work well
#multiple options 

#transform the data to get rid of the tail and make the data look normal and then scale
#can do this by raising all features to a power less than 1 or if the feature has a very strong tail log each feature
#or 
#you could bucketize a feature  

#bucketizing can also be useful when a feature is multimodal 

#Another approach for dealing with a multimodal feature is to add a feature for each of the modes
#then use a radial basis function called rbf_kernal
#in our case the house medain age has 2 peaks, one at 35years
from sklearn.metrics.pairwise import rbf_kernel
age_similar_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)

#Note that while you scale and transform the data for the model, we can make more sence of it, if its in its origional scale
#all of the transformers also have a inverse_transform to do this for us 

#Sklearn has TransformedTargetRegressor that does this whole proccess for us
#need to specify the reression model and the type of scaler
#ex this model uses median income to predict median house value
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
model = TransformedTargetRegressor(LinearRegression(), transformer=StandardScaler())
model.fit(housing[["median_income"]], housing_labels)
#predictions = model.predict(new_data)


#Custom transformers/scalers 
#sklearn has a lot of scalers but you will need to  create your own if none fit your needs
#custom transforms can also combine specific attributes 

#ex creating a tranformer to scale by log for our population feature that is skewed
from sklearn.preprocessing import FunctionTransformer
log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing[["population"]])

#trainable custom transfomers require making a class
#this allows to fit the tranformer to a parameter then transform  using that fitted paramter
#require fit, transform, and fit_transorm(TransformerMixin)





#Pipelines - can do whole procces of cleaning and transforming data in one step
#pipeline is like a chain of transformations that are applied in order
#usefull when continously getting new datat that needs to be cleaned and transformed and then fed into the model
#they ahve fit, transform, and fit_transform methods, and can have predict if thats the last arugment (ie whole model in 1)
#the seqence of steps will be in the order than is typed in the pipeline

from sklearn.pipeline import Pipeline, make_pipeline

#simple pipline that fills in missing values w/ median and then standard scales the data
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())
])

#pipeline if you dont want to name each transfomrer
num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

#now we can use the pipeline to transform the data in one quick step
housing_num_prepared = num_pipeline.fit_transform(housing_num)

#pipelines out pusts are 2d numpy arrays, so we need to convert the data back into a dataframe
df_housing_num_prepared = pd.DataFrame(housing_num_prepared, columns=num_pipeline.get_feature_names_out(), index=housing_num.index)

#can also use column transformer to apply different transformations to different columns
#because so far we have to use differnt transformers or a pipeline to handle the catigorical data and numerical data
from sklearn.compose import ColumnTransformer

num_attribes = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]
cat_attribes = ["ocean_proximity"]

#pipeline that transforms and then scales the catigorical data
cat_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder())

#full pipeline that applies the catigorical pipeline to the catigorical data and the num pipeline to the numerical data
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribes),
    ("cat", cat_pipeline, cat_attribes)
])

#however, selecting all of the columns of certain types could be apain, so we can use a selector to do it for us
#make_column_selector(dtype_include=np.number) selects all the numerical columns, very similar to the full_pipeline above
#leaving the names out of the first entry will automatically name the transformers pipeline_1 and pipeline_2 instead of num and cat
from sklearn.compose import make_column_selector, make_column_transformer

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object)))


housing_prepared = preprocessing.fit_transform(housing)

#left out one transformer that the book covers (cluseting a feature) here is the completed pipeline from the book

from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin


class ClusterSimilarity(BaseEstimator, TransformerMixin):


    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10,
                              random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())
preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline) 


housing_prepared = preprocessing.fit_transform(housing)
# extra code – shows that we can get a DataFrame out if we want
# housing_prepared_fr = pd.DataFrame(
#     housing_prepared,
#     columns=preprocessing.get_feature_names_out(),
#     index=housing.index)




