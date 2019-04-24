import os
import tarfile
import pandas as pd
import matplotlib.pyplot as plt

PATH = "/media/ad5146/Anthony D/school/2018-19/spring/csc 403 (ML)/CSC403_FinalProject/historical-hourly-weather-data/"
FILE_NAMES = {"humidity.csv", "pressure.csv", "temperature.csv",
              "weather_description.csv", "wind_direction.csv", "wind_speed.csv"}

# reads in all csv's of data
def load_weather_data():
    result_df = pd.DataFrame()
    
    for x in FILE_NAMES:
        
        df = pd.read_csv(PATH + x)
        
        df = df[["Portland"]]
        
        result_df = pd.concat([result_df, df])
	
        print(x, "was read in successfully")
        print(result_df)
        
    return result_df

def main():
    data = load_weather_data()
    print("data")
    print(data)
    
    portland = data[2]
    portland_df = pd.DataFrame(portland)
    print(portland_df)

main()
