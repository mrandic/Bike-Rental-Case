# include all libraries needed for the project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_processing as dpr
import data_plotting as dpl
import ml_processing as ml
import xgboost as xgb
from dateutil.parser import parse
from sklearn.cluster import KMeans
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


# import raw datasets
hubway_trips_df, hubway_stations_df, weather_df, zip_code_gps_df = dpr.importData()

# merge all datasets into master data set
master_df = dpr.createMasterDataSet(hubway_trips_df, hubway_stations_df, weather_df, zip_code_gps_df)

# Create additional features derived from exploratory analysis
# Complete list of selected features will be elaborated in Step3
# This is a final product of features to be used for data preparation in ML process
master_df = dpr.createFeatures(master_df)

# Create subset of variables from master dataset suitable for modeling
feature_set = dpr.featureSubset(master_df)

# Adapt column names to standard naming convention
feature_set = dpr.renameColumns(feature_set)

# Drop events with missing location values
feature_set = feature_set.drop(feature_set[feature_set['latitude'].isnull()].index)
feature_set = feature_set.drop(feature_set[feature_set['longitude'].isnull()].index)

# Impute values for missing variables with data
feature_set.loc[feature_set.age.isnull(), "age"] = feature_set.age.mean()
feature_set.loc[feature_set.gender.isnull(), "gender"] = 'NULL'

# Assign 'Category' type for selected variables
feature_set = dpr.setFeatureCategoryType(feature_set)

# Peform K-Means clustering on location data and build 5-category feature
kmeans = KMeans(5)
clusters = kmeans.fit_predict(feature_set[['latitude','longitude']])
feature_set['location_cluster'] = kmeans.predict(feature_set[['latitude','longitude']])
feature_set["location_cluster"] = feature_set["location_cluster"].astype('category')

# Perform One-Hot encoding on category variables
category_features_ohc = pd.get_dummies(feature_set[[
                                                 'subsc_type',
                                                 'gender',
                                                 'weather_event',
                                                 'location_cluster'
                                                ]])

# Remove category variables from feature set
category_features_ohc = category_features_ohc.drop(columns=['gender_NULL', 'weather_event_None'])

# Remove category variables from feature set
feature_set = feature_set.drop(columns=['staton_municipality',
                                        'subsc_type',
                                        'gender',
                                        'avg_dew_point_f',
                                        'bike_freq_use_range',
                                        'bike_avg_dur_range',
                                        'weather_event',
                                        'station_status',
                                        'trip_status',
                                        'zip_code',
                                        'location_cluster',
                                        'year',
                                        'latitude',
                                        'longitude'
                                                ])

# Join OHC features to final feature set
# Focus on trip durations below 1000s
# After many trials, trip durations over 1000s decreased model performance significantly
feature_set_ml = feature_set.join(category_features_ohc)
feature_set_ml = feature_set_ml[(feature_set_ml["duration"] > 0) & (feature_set_ml["duration"] <= 1500)]

# Picking out the relevant attributes for regression modelling
# Top 10 correlated variables are chosen for final feature set
# As already confirmed through exploratory analysis, 'casual' users have showed highest influence on trip duration
# Casual users probably pay one-time fee and tend to maximize trip duration for their spent money.
columns = ml.calculateTopCorrelatedColumns(feature_set_ml, 10)

# Large duration numbers may affect the absolute numbers of the regression model.
# To prevent this, target variable will be log-normalized.
# At the end of ML pipeline, reverse operation will be done in order to retrieve original values.
feature_set_ml['duration'] = np.log(feature_set_ml['duration'])

# Define feature and target set
X = feature_set_ml[columns]
Y = X['duration'].values
X = X.drop('duration', axis = 1).values

# Split dataset into train (70%) and test (30%) ratios.
X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.30, random_state=42)

# Train challenger models with their default parameters
# Following regression models are included in this benchmark:
# LinearRegression
# Lasso
# ElasticNet
# KNeighborsRegressor
# DecisionTreeRegressor
# XGBRegressor
# GradientBoostingRegressor

# The differing scales of the raw data may impact chosen algorithms.
# Part of a requirement for a standardised data set is for each attribute to have a mean value of zero and a standard deviation of 1.
# This is implemented within provided function.
# Cross-validation is used to validate performance of algorithms.
# Model with maximun negative MSE and minimum standard deviation is chosen as a winning model

# ml.modelChallengerDefaultParams(X_train, Y_train)

# Perform winning model auto hyperparameter tuning
# ml.selectBestModelParams(dict(n_estimators=np.array([50,100,150,200])), xgb.XGBRegressor(random_state=21), X_train, Y_train )

# Train model with best parameters
model = xgb.XGBRegressor(random_state=21, n_estimators=50, alpha=0.2)
model, scaler = ml.trainModel(model, X_train, Y_train)

# Run model on unseen test dataset and output MSE
predictions = ml.testModelMSE(model, scaler, X_test, Y_test)

# Comparison of predicted vs real data (scaled)
ml.compareResultsLog(predictions, Y_test)

# Reverse scaled predictions to real values and display comparison
ml.compareResultsActual(predictions, Y_test)
