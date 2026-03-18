from matplotlib import pyplot as plt
import csv
import numpy as np

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

'''
Implement a perceptron algorithm manually in Python 
'''
class Perceptron:
    def __init__(self, l_rate = 0.1, i = 10):
        self.l_rate = l_rate #dont input learning rate because it should be 0.1
        self.i = i
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        k_samples, k_features = X.shape

        #initalizing weights to be randomly between -0.5, and 0.5
        self.weights = np.random.uniform(-0.5, 0.5, size = (k_features)) 
        
        #Bias term for implementation 
        self.bias = 0
       
        for _ in range(self.i):
            for idx, x_i in enumerate(X):
                linear_o = np.dot(x_i, self.weights) + self.bias
                y_predict = self.activation_func(linear_o)

                #manually coding the weight updates --> is there a formula to update the weights? cause if i do this its always 0
                update = self.l_rate * (y[idx] - y_predict)

                self.weights += update * x_i
                self.bias += update 
                
                '''
                CHECKING FOR ERRORS:
                if update != 0:
                    print(f"_ : {_} | Update: {update} | Weights: {self.weights}") '''

                
    def prediction(self, X):
        linear_o = np.dot(X, self.weights) + self.bias
        y_perdict = self.activation_func(linear_o)
        return y_perdict

    #ACTIVATION FUNCTION --> manually coded
    def _unit_step_func(self, x):
        return np.where(x>= 0, 1, 0)


def b(iterations):
    X, y = [], []

    #get data and organize it
    with open('data/WeatherData_Q3.csv', 'r') as csv_f:
        csv_reader = csv.DictReader(csv_f)

        for row in csv_reader:
            
            #using humidity and temperature as input
            data = [float(row['temp']),float(row['humid'])]
            X.append(data)

            y.append(int(row['rain']))
    
    X, y = np.array(X), np.array(y)

    
    #split the dataset into training set and test set (split at 15)
    X_train, X_test = X[:14], X[15:]
    y_train, y_test = y[:14], y[15:]
    

    ''' 
    #Split at 5
    X_train, X_test = X[:4], X[5:]
    y_train, y_test = y[:4], y[5:]
    '''
    '''
    #split at 18
    X_train, X_test = X[:17], X[18:]
    y_train, y_test = y[:17], y[18:]
    '''
    
    #create a perceptron
    p = Perceptron(i = iterations)

    p.fit(X_train,y_train)
    perdictions= p.prediction(X_test)
    
    #REPORT THE TRAINING AND ACCURACY AFTER TRAINING
    print(f'Perceptron accuracy classification is: {accuracy(y_test, perdictions)}')
   
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0,1],label= "No Rain", c='blue', marker='s')
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1,1],label= "Rain",c='red', marker='o')

    #calc the line
    x0_1 = np.amin(X_train[:,0])
    x0_2 = np.amax(X_train[:,0])
    x1_1 = (-p.weights[0] *x0_1 - p.bias) / p.weights[1]
    x1_2 = (-p.weights[0] *x0_2 - p.bias) / p.weights[1]

    ax.plot([x0_1,x0_2], [x1_1,x1_2], 'k', label = "Decision Boundary")

    #set limits and labels
    ymin = np.amin(X_train[:,1])
    ymax = np.amax(X_train[:,1])
    ax.set_ylim([ymin-3,ymax+3])
    plt.title("Weather Perceptron Plotted")
    plt.xlabel("Temperature")
    plt.ylabel("Humidity")
    plt.legend()
    plt.show()


#for predicting accuracy 
def accuracy(y_true, y_predict):
    accuracy = np.sum(y_true == y_predict) / len(y_true)
    return accuracy


#runs program for Problem 3 part a
a()

#train on number of 'iterations'
iterations = 500
b(iterations)




