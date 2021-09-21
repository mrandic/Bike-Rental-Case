import numpy as np
import pandas as pd
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
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

def calculateTopCorrelatedColumns(feature_set, n):
    """
    Calculate Pearson correlation for top 10 most correlated features with target variable
    :param feature_set: Feature set
    :param n: Number of top correlated features

    :return: Columns with highest correlation
    """
    # Run Pearson correlation and select top 10 correlated features
    correlation = feature_set.corr(method='pearson')
    columns = correlation.nlargest(n, 'duration').index
    return columns



def modelChallengerDefaultParams(X_train, Y_train):
    """
    Train challenger models with their default parameters
    :param X_train: Observation set
    :param Y_train: Target set

    :return: Information on winning model
    """

    # Define pipeline collection and store regression models
    # Apply scaler for each pipeline
    pipelines = []
    pipelines.append(('ScaledXGBM', Pipeline([('Scaler', StandardScaler()),('XGBM', xgb.XGBRegressor())])))
    pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])))
    pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
    pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
    pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
    pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
    pipelines.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))


    # Cross-validation (in 5 fold splits) is used to validate performance of algorithms
    # Prints performance information for each model
    results = []
    names = []
    for name, model in pipelines:
        kfold = KFold(n_splits=5, random_state=42, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='neg_mean_squared_error')
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)


def selectBestModelParams(param_grid, model, X_train, Y_train):
    """
    Performs hyper-parameter tuning for the model, from given criteria
    :param param_grid: Set of parameter ranges
    :param model: Initialized model
    :param X_train: Observation set
    :param Y_train: Target set

    :return: Information on optimal parameter configuration
    """

    # Initialize scaler and cross-validation for the model
    scaler = StandardScaler().fit(X_train)
    rescaledX = scaler.transform(X_train)
    kfold = KFold(n_splits=5, random_state=21, shuffle=True)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kfold)
    grid_result = grid.fit(rescaledX, Y_train)

    # extract evaluation parameters
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


def trainModel(model, X_train, Y_train):
    """
    Performs model training after being supplied with optimal parameters
    :param model: Initialized and configured model
    :param X_train: Observation set
    :param Y_train: Target set

    :return: model and scaler for the model to be used in evaluation
    """
    scaler = StandardScaler().fit(X_train)
    rescaled_X_train = scaler.transform(X_train)
    #model = GradientBoostingRegressor(random_state=21, n_estimators=400)
    model.fit(rescaled_X_train, Y_train)
    return model, scaler


def testModelMSE(model, scaler, X_test, Y_test):
    """
    Performs model scoring on unseen data
    :param model: Trained model
    :param X_test: Observation set
    :param Y_test: Target set

    :return: Predictions from the model
    """
    # transform the validation dataset
    rescaled_X_test = scaler.transform(X_test)
    predictions = model.predict(rescaled_X_test)
    print (mean_squared_error(Y_test, predictions))
    return predictions


def compareResultsLog(predictions, Y_test):
    """
    Compares and lists predicted vs actual trip durations on scaled level
    :param predictions: Model output
    :param Y_test: Target set

    :return: Information on predicted vs actual trip durations
    """
    compare = pd.DataFrame({'Prediction': predictions, 'Test Data' : Y_test})
    print(compare.head(10))


def compareResultsActual(predictions, Y_test):
    """
    Compares and lists predicted vs actual trip durations on reversed, real trip durations
    :param predictions: Model output
    :param Y_test: Target set

    :return: Information on predicted vs actual trip durations
    """
    actual_y_test = np.exp(Y_test)
    actual_predicted = np.exp(predictions)
    diff = abs(actual_y_test - actual_predicted)

    compare_actual = pd.DataFrame({'Expected Duration': actual_y_test, 'Predicted Duration' : actual_predicted, 'Difference' : diff})
    compare_actual = compare_actual.astype(int)
    print(compare_actual.head(10))