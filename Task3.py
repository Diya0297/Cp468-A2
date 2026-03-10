from matplotlib import pyplot as plt
import csv

'''
A smart farming company is developing an AI-based weather prediction model to help farmers make better decisions about irrigation and crop protection decisions. They want to use historical weather data (temperature and humidity) to predict whether it will rain.
To achieve this, they will train a Perceptron-based AI model to classify weather conditions into two categories:
 y = 0 (No Rain)
 y = 1 (Rain)
The company collects data from sensors and stores it in WeatherData.csv, which contains 20 recorded instances of weather conditions with:
 x1 = Temperature (scaled between 0 and 1)
 x2 = Humidity (scaled between 0 and 1)
 y = Rain (0 or 1)
The company needs to analyze this dataset and evaluate the performance of a Perceptron model in predicting rainfall.
'''

'''
a)
Plot the weather data on a 2D graph where:
-  No Rain (y = 0) is represented as blue squares. o Rain (y = 1) is represented as red circles.
-  Include labels, a title, and a legend for clarity.
'''
def a():
    NoRain_x1 = []
    NoRain_y1 = []
    Rain_x2 = []
    Rain_y2 = []

    with open('data/WeatherData_Q3.csv', 'r') as csv_f:
        csv_reader = csv.DictReader(csv_f)

        for row in csv_reader:
            
            
            if row['rain'] == '0':
                NoRain_x1.append(float(row['temp']))
                NoRain_y1.append(float(row['humid']))
            else:
                Rain_x2.append(float(row['temp']))
                Rain_y2.append(float(row['humid']))

    plt.scatter(NoRain_x1,NoRain_y1, label= "No Rain", c='blue', marker='s')
    plt.scatter(Rain_x2,Rain_y2,label= "Rain",c='red', marker='o')

    plt.title("Weather Data Scatter plot for Rain vs No Rain")
    plt.xlabel("Temperature")
    plt.ylabel("Humidity")
    plt.legend()

    plt.show()

a()
