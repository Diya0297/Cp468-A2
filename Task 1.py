import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from collections import Counter

df = pd.read_csv("/Users/aasthapunj/Desktop/cp468/a02/CustomerDataset_Q1.csv")

#(a)
y0 = df[df['c'] == 0]
y1 = df[df['c'] == 1]
plt.scatter(y0['x1'], y0['x2'], color='blue', label='Not interested')
plt.scatter(y1['x1'], y1['x2'], color='red', label='Interested')
#bbox_to_anchor=(1,0.5) has been used because the identifiers table was on top of the graph for this plot
#so it was making it difficult to interpret so I have pushed it outside the graph so the stacked bar chart is clearly distinguishable.
plt.legend(bbox_to_anchor=(1,0.5))  #legend for differentiating colors
plt.xlabel("Average amount spent per purchase (in dollars)")  #x-label
plt.ylabel("Frequency of purchases per month")  #y-label
plt.tight_layout()
plt.show()

#(b)
#starting with defining a function for Euclidean distance
def euclidean_distance(x,y):
    return np.sqrt(np.sum((np.array(x)-np.array(y)) ** 2))


def fnKNN(df, new_point, k):
    #to calculate and keep track of distance of new point from every point already on the graph (i.e. customer)
    #this will also store the class i.e. 0 or 1 (labelled as y in the dataset)
    #therefore, this array distance will store [distance, class label]
    distance = []  
    for i in range(len(df)):  #i is every row in the csv file
        point = [df.iloc[i]['x1'], df.iloc[i]['x2']]  #iloc will get the value of x1, x2 and c from every row(i) in the file
        y_label = df.iloc[i]['c']
        d = euclidean_distance(point, new_point)
        distance.append((d,y_label))

    distance.sort(key=lambda x:x[0]) #sorting distance array by euclidean distance which is at the 0th index of the subarray
    knn = distance[:k] #k nearest neighbours (takes first k elements from the sorted array)
    knn_labels = [i[1] for i in knn] #getting class labels (0 or 1) for all chosen neighbours
    predicted = Counter(knn_labels).most_common(1)[0][0]  #getting the majorit vote from the number of labels to choose the class (0 or 1)
    return int(predicted)

# print(fnKNN(df,[0.5,0.5],3))

#(c)
#Use a fixed random seed (e.g., 42) for reproducibility.
np.random.seed(42) #shuffling the dataset but keep the same seed for reproducibility
shuffled_dataset = df.sample(frac=1, random_state=42) #frac=1 shuffles the whole dataset but keeps the same number of rows

def assess_accuracy(df, train_test_size, k):
    split = int(len(df)*train_test_size)
    train = df.iloc[:split]
    test = df.iloc[split:]
    
    correct_prediction = 0 #this will check how many prediction match the actual y value (class label) given in the dataset
    for i in range(len(test)):  #looping through test set only
        point = [test.iloc[i]['x1'], test.iloc[i]['x2']]
        y_label = test.iloc[i]['c']

        predicted = fnKNN(train, point, k)
        if predicted == y_label: #if prediction is correct the counter for correct_prediction increments by 1
            correct_prediction += 1

    accuracy = correct_prediction/len(test) #correct predictions / total samples in the dataset

    return accuracy

#this is for k=1
print("Accuracy for 80% split (16 train, 4 test) when k=1: ", assess_accuracy(shuffled_dataset, 0.8, 1))
print("Accuracy for 60% split (12 train, 8 test) when k=1: ", assess_accuracy(shuffled_dataset, 0.6, 1))
print("Accuracy for 50% split (10 train, 10 test) when k=1: ", assess_accuracy(shuffled_dataset, 0.5, 1))
print()

#(d)
#this is for k=2
print("Accuracy for 80% split (16 train, 4 test) when k=2: ", assess_accuracy(shuffled_dataset, 0.8, 2))
print("Accuracy for 60% split (12 train, 8 test) when k=2: ", assess_accuracy(shuffled_dataset, 0.6, 2))
print("Accuracy for 50% split (10 train, 10 test) when k=2: ", assess_accuracy(shuffled_dataset, 0.5, 2))
print()

#this is for k=3
print("Accuracy for 80% split (16 train, 4 test) when k=3: ", assess_accuracy(shuffled_dataset, 0.8, 3))
print("Accuracy for 60% split (12 train, 8 test) when k=3: ", assess_accuracy(shuffled_dataset, 0.6, 3))
print("Accuracy for 50% split (10 train, 10 test) when k=3: ", assess_accuracy(shuffled_dataset, 0.5, 3))
print()

#this is for k=4
print("Accuracy for 80% split (16 train, 4 test) when k=4: ", assess_accuracy(shuffled_dataset, 0.8, 4))
print("Accuracy for 60% split (12 train, 8 test) when k=4: ", assess_accuracy(shuffled_dataset, 0.6, 4))
print("Accuracy for 50% split (10 train, 10 test) when k=4: ", assess_accuracy(shuffled_dataset, 0.5, 4))