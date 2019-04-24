import os
import tarfile
import pandas as pd
import matplotlib.pyplot as plt

PATH = "/home/jb7656/MachineLearning/Group Project/Datasets/historical-hourly-weather-data/"
FILE_NAMES = {"humidity.csv", "pressure.csv", "temperature.csv",
"weather_description.csv", "wind_direction.csv", "wind_speed.csv"}
# reads in all csv's of data
def load_weather_data():
    result_df = pd.DataFrame()
    index = 0
    for x in FILE_NAMES:
        df = pd.read_csv(PATH + x)
        
        df = df[["Portland"]]

        if(index == 0):
            result_df = pd.concat([result_df, df])
            
        else:
            result_df.insert(index, x, df, allow_duplicates = False)

        index = index + 1
        print(x, "was read in successfully")
        print(result_df)   
    return result_df

def main():
    data = load_weather_data()
    print("data")
    data.columns = ['weather_description', 'humidity', 'pressure', 'wind_speed',
       'temperature', 'wind_direction']
    print(data)
    print(data.columns)
    
    #portland = data[2]
    #portland_df = pd.DataFrame(portland)
    #print(portland_df)

main()

