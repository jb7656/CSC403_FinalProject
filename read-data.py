import os
import tarfile
import pandas as pd
import matplotlib.pyplot as plt

PATH = "/home/jb7656/MachineLearning/Group Project/Datasets/historical-hourly-weather-data/"
FILE_NAMES = {"humidity.csv", "pressure.csv", "temperature.csv",
              "weather_description.csv", "wind_direction.csv", "wind_speed.csv"}
def load_weather_data():
    #need to read in all csv's of data
    #return pd.read_csv(PATH + FILE_NAME)
    csv_list = []
    for x in FILE_NAMES:
        csv_list.append(pd.read_csv(PATH + x))
        print(x, "was read in successfully")
    return csv_list 
        

def main():
    data = load_weather_data()
    portland = data[2]
    portland_df = pd.DataFrame(portland)
    print(portland_df)
    

    

main()
