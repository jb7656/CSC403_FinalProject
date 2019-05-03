import os
import tarfile
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split

PATH = os.getcwd() + "/data/" # "/home/jb7656/MachineLearning/Group Project/Datasets/historical-hourly-weather-data/"
FILE_NAMES = ["weather_description.csv","humidity.csv", "pressure.csv", "temperature.csv",
              "wind_speed.csv", "wind_direction.csv" ]
Classification_names = ["mist", "broken clouds", "sky is clear", "light rain", "few clouds",
                        "moderate rain", "haze", "heavy intensity rain", "heavy snow",
                        "thunderstorm", "freezing rain"]

# reads in all csv's of data
def load_weather_data():
    result_df = pd.DataFrame()
    index = 0
    for x in FILE_NAMES:
        df = pd.read_csv(PATH + x)
        
        df = df[["Portland"]]

        if(index == 0):
            result_df = pd.concat([result_df, df])
            result_df.columns = [x]
            
        else:
            result_df.insert(index, x, df, allow_duplicates = False)

        index = index + 1
        print(x, "was read in successfully")
        #print(result_df)
    return result_df





def direction_num_to_string(num):
    val = int(int(num)/45)
    #arr=["N","NE","E","SE","S","SW","W","NW"]
    arr = [0,1,2,3,4,5,6,7]
    return arr[val%8]

#need to assign specific categorical number for weather description,
#as there are too many unique categories for the tree to find,
#so combining some will reduce error
def convert_desc_to_cat(argument):
    if(argument == "mist"):
        return 0
    elif(argument == "fog"):
        return 0
    elif(argument == "broken clouds"):
        return 1
    elif(argument == "scattered clouds"):
        return 1 #same as broken clouds so class is same
    elif(argument == "sky is clear"):
        return 2      
    elif(argument == "light rain"):
        return 3
    elif(argument == "drizzle"):
        return 3
    elif(argument == "light intensity drizzle"):
        return 3
    elif(argument == "few clouds"):
        return 4  
    elif(argument == "overcast clouds"):
        return 4
    elif(argument == "moderate rain"):
        return 5
    elif(argument == "light intensity shower rain"):
        return 5
    elif(argument == "haze"):    
        return 6
    elif(argument == "smoke"):
        return 6
    elif(argument == "dust"):
        return 6
    elif(argument == "heavy intensity rain"):
        return 7
    elif(argument == "very heavy rain"):
        return 7 #same as heavy intensity rain
    elif(argument == "heavy snow"):
        return 8
    elif(argument == "light snow"):
        return 8
    elif(argument == "snow"):
        return 8
    elif(argument == "proximity thunderstorm"):
        return 9
    elif(argument == "thunderstorm"):
        return 9
    elif(argument == "thunderstorm with light rain"):
        return 9
    elif(argument == "freezing rain"):
        return 10
    elif(argument == "sleet"):
        return 10 #same as freezing rain
    else:
        print(argument, "found uncategorized instance")
        return -1
    
    
def main():
    data = load_weather_data()

    # removes the first NAN row indexed 0
    data = data.iloc[1:,]
    # Converts wind_direction column numbers to strings.
    for a in data['wind_direction.csv']:
       if(math.isnan(a)== False): 
          #print(a,direction_num_to_string(a))    # before conversion
          a = direction_num_to_string(a)
          #print(a)                               # after conversion


    dataY = data['wind_direction.csv']
    dataX = data.drop(columns='wind_direction.csv')

    weather_desc = data["weather_description.csv"]
    x = 0
    for value in weather_desc:
        weather_desc[x] = convert_desc_to_cat(value)
        x = x + 1
    weather_desc[x] = convert_desc_to_cat("broken clouds")
    #print(weather_desc)

    data["weather_description.csv"] = weather_desc
    #data.insert(5, dataY, "wind_direction")

    
    data = data.drop("weather_description.csv", axis = 1)
    data = data.fillna(method = 'bfill')
    print(data.head)
    print(data.columns)

    weather_desc = weather_desc.astype('int')
    weather_desc = weather_desc[1:]

    #split the data into a training set and testing set
    data_train, data_test, label_train, label_test = train_test_split(data, weather_desc, test_size = 0.1)


    #Data has been processed at this point and is all numerical
    depth = 5
    while depth < 12: #testing various depths and scoring their classification performance
        tree_clf = DecisionTreeClassifier(max_depth = depth, min_samples_split = 10)
        tree_clf.fit(data_train, label_train)
        print("(Gini) depth: ", depth, " score: ", (100 * tree_clf.score(data_test, label_test)),"%")
        depth = depth + 1

    depth = 5
    while depth < 12: #testing various depths using entropy rather than gini and scoring performance
        tree_clf = DecisionTreeClassifier(max_depth = depth, criterion = "entropy", min_samples_split = 10)
        tree_clf.fit(data_train, label_train)
        print("(Entropy) depth: ", depth, " score: ", (100 * tree_clf.score(data_test, label_test)), "%")
        depth = depth + 1
    #use this export statement to create visualized tree
    #export_graphviz(tree_clf, out_file = "tree_path.dot",
                   #rounded = True, filled = True, class_names = Classification_names)
    

main()

