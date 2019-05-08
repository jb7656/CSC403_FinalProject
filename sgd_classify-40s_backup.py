def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

import os
import tarfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_predict, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score

PATH = os.getcwd() + "/data/" # "/home/jb7656/MachineLearning/Group Project/Datasets/historical-hourly-weather-data/"
FILE_NAMES = ["humidity.csv", "pressure.csv", "temperature.csv", "wind_direction.csv", "wind_speed.csv", "weather_description.csv"]

# reads in all csv's of data
def load_weather_data():
    result_df = pd.DataFrame()

    for i in range(0, len(FILE_NAMES)):

        f = FILE_NAMES[i]
        df = pd.read_csv(PATH + f)
        df = df[["Portland"]]

        if(i == 0):
            result_df = pd.concat([result_df, df])
            result_df.columns = [f[:len(f) - 4]]
        else:
            result_df.insert(i, f[:len(f) - 4], df, allow_duplicates = False)

    return result_df

def clean(data):

    data = data.copy()
    data.dropna(inplace=True) # drop any rows with a missing value

    # weather description modifications
    # remove some descriptions
    data.where(data["weather_description"] != "mist", inplace=True)
    data.where(data["weather_description"] != "fog", inplace=True)
    data.where(data["weather_description"] != "haze", inplace=True)
    data.where(data["weather_description"] != "smoke", inplace=True)
    data.where(data["weather_description"] != "light snow", inplace=True)
    data.where(data["weather_description"] != "snow", inplace=True)
    data.where(data["weather_description"] != "proximity thunderstorm", inplace=True)
    data.where(data["weather_description"] != "dust", inplace=True)
    data.where(data["weather_description"] != "freezing rain", inplace=True)
    data.where(data["weather_description"] != "heavy snow", inplace=True)
    data.where(data["weather_description"] != "sleet", inplace=True)
    data.dropna(inplace=True)
    # combine some descriptions
    y = data["weather_description"]
    y.replace("sky is clear", "clear", inplace=True)
    y.replace("light rain", "rain", inplace=True)
    y.replace("moderate rain", "rain", inplace=True)
    y.replace("heavy intensity rain", "rain", inplace=True)
    y.replace("light intensity drizzle", "rain", inplace=True)
    y.replace("very heavy rain", "rain", inplace=True)
    y.replace("thunderstorm", "rain", inplace=True)
    y.replace("thunderstorm with light rain", "rain", inplace=True)
    y.replace("drizzle", "rain", inplace=True)
    y.replace("overcast clouds", "clouds", inplace=True)
    y.replace("broken clouds", "clouds", inplace=True)
    y.replace("few clouds", "clouds", inplace=True)
    y.replace("scattered clouds", "clouds", inplace=True)
    # encode
    encoder = LabelEncoder()
    # put back
    data.drop("weather_description", axis=1, inplace=True)
    data["weather_description"] = encoder.fit_transform(y)
    data.reset_index(inplace=True)

    # standardize all the numerical values
    X = data.drop("weather_description", axis=1)
    std_scaler = StandardScaler()
    X_scaled = std_scaler.fit_transform(X)
    # put back
    result_data = DataFrame(X_scaled, columns=X.columns)
    result_data["weather_description"] = data["weather_description"].copy()

    return result_data, encoder

def split(data):

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data, data["weather_description"]):
        train_set = data.loc[train_index]
        test_set = data.loc[test_index]

    X_train = train_set.drop("weather_description", axis=1).values
    y_train = train_set["weather_description"].copy().values
    X_test = test_set.drop("weather_description", axis=1).values
    y_test = test_set["weather_description"].copy().values

    return X_train, y_train, X_test, y_test

def train(X_train, y_train):
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train)
    return sgd_clf

def print_scores(sgd_clf, X, y):
    y_pred = cross_val_predict(sgd_clf, X, y, cv=3)
    print("accuracy")
    print(cross_val_score(sgd_clf, X, y, cv=3, scoring="accuracy"))
    print("precision")
    print(precision_score(y, y_pred, average=None))
    print("recall")
    print(recall_score(y, y_pred, average=None))
    print("f1")
    print(f1_score(y, y_pred, average=None))

def main(data=None):

    if data is None:
        data = load_weather_data()

    clean_data, encoder = clean(data)
    X_train, y_train, X_test, y_test = split(clean_data)
    sgd_clf = train(X_train, y_train)

    print("---Training set---")
    print_scores(sgd_clf, X_train, y_train)
    print("---Test set---")
    print_scores(sgd_clf, X_test, y_test)

    return data, sgd_clf, encoder