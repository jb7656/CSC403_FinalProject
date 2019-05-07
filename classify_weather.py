import os
import tarfile
import pandas as pd
import matplotlib.pyplot as plt

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
    data = data.dropna()
    X = data.drop("weather_description", axis=1)
    y = data["weather_description"].copy()
    return X, y

def main():
    data = load_weather_data()
    X, y = clean(data)

    print("=====================================")
    print(X)
    print("=====================================")
    print(y)
    print("=====================================")