import pandas as pd
import numpy as np
from collections import Counter

mushrooms_df = pd.read_csv('data/mushrooms.csv')

def encode_labels(df):
    import sklearn.preprocessing
    encoder = {}
    for col in df.columns:
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(df[col])
        df[col] = le.transform(df[col])
        encoder[col] = le
    return df, encoder    

mushrooms_encoded_df, encoder = encode_labels(mushrooms_df)

X_df = mushrooms_encoded_df.drop('class', axis=1)  # attributes
X_df = mushrooms_encoded_df.drop('veil-type', axis=1)  # attributes
y_df = mushrooms_encoded_df['class']  # classes
X_array = X_df.to_numpy()
y_array = y_df.to_numpy()

print('X =', X_array)
print('y =', y_array)

def train_test_split(X, y, test_size=0.2):
    """
    Shuffles the dataset and splits it into training and test sets.
    
    :param X
        attributes
    :param y
        classes
    :param test_size
        float between 0.0 and 1.0 representing the proportion of the dataset to include in the test split
    :return
        train-test splits (X-train, X-test, y-train, y-test)
    """
    
    
    # This is how many rows we want our train set to be
    len_train = int((1-test_size) * len(X))

    # Merging the class and attribute columns
    data_array = np.concatenate((X, y[:,None]), axis = 1)
    # Shuffling the rows in data_array
    np.random.shuffle(data_array)
    
    # Slicing the data array such that it fits the return values
    # Attribute values
    X_train = data_array[:len_train,1:-1]
    X_test = data_array[len_train:,1:-1] 
    # Class values
    y_train = data_array[:len_train,0]
    y_test = data_array[len_train:,0] 
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, 0.33)

class Node:
    """
    Class to represent a node in the tree which stores a list of children nodes and the data beloning to that node.
    Data is a subset of the whole data_array.
    """
    def __init__(self, data=''):
        self.data = data
        self.children = []

    def isEmpty(self):
        return (self.data is None)

    def setData(self,data):
        self.data = data
    
    def getData(self):
        return self.data

    def addChild(self, child):
        self.children.append(child)

    def getChildren(self):
        return self.children

def giniCoefficient(attribute_count):
    """
    Compute Gini coefficient of array of values

    :param attribute_count
    Row of T/F values given a unique attribute-value in a column of data_array

    :return gini 
    Gini value of this row
    """
    temp = 0
    s = sum(attribute_count)
    for value in attribute_count:
        temp = (- value/s)**2
    gini = 1 - temp
    return gini

def bestSplit(X_train,y_train):
    # Number of attributes
    n = len(X_train[0]) 
    d = len(X_train)
    classes = np.unique(y_train)
    data_array = np.concatenate((X_train, y_train[:,None]), axis = 1)
    # List of calculated gini splits for all attributes
    gini_split_value = []
    
    # Iterate over all attributes to calculate their gsplits
    #i = 2
    for i in range(n):
        # Get i-th column of X_train and combine it with class column
        column_target = np.vstack((X_train[:,i], y_train))
        unique_attribute = np.unique(X_train[:,i])
        
        # Get count of each ocurrence given an attribute and y_train
        # Row = attributes, Columns = count for class T/F
        unique_attribute_count = np.zeros(shape=(len(unique_attribute),len(classes)))
        # Shady stuff, if you have time FIX
        for row in column_target.T:
            if (row[1]==1):
                unique_attribute_count[row[0],0] +=1
            else: 
                unique_attribute_count[row[0],1] +=1

        #Calculating the gini values for each unique attribute value
        gini_unique_attribute = np.zeros(shape=(len(unique_attribute_count),))
        j = 0
        for row in unique_attribute_count:
            gini_unique_attribute[j] = giniCoefficient(row)
            j += 1

        # Calculating the split value
        row_sum = np.sum(unique_attribute_count, axis = 1)
        # Count numbers of occurences for all attributes
        tot_sum = np.sum(unique_attribute_count)
        if (len(gini_unique_attribute)==1):
            gini_split_value.append(gini_unique_attribute[0])

        else:
            sum_gini = 0
            for x,y in zip(row_sum,gini_unique_attribute):
                sum_gini += (y/tot_sum)*x
            gini_split_value.append(sum_gini)
    # Get maximum attribute index
    min_gini= gini_split_value.index(min(gini_split_value))
    return min_gini

def stopCondition(classes_val):
    counts = classes_val[1]
    len_counts = len(counts)
    if (len_counts==1): 
        return True
    else:
        return False

def partitionDataset(data, split):
    indices = np.argsort(data[:, split])
    arr_temp = data[indices] 
    subset_dataarray = np.array_split(arr_temp, np.where(np.diff(arr_temp[:,split])!=0)[0]+1)
    # Removing split column from subset_data
    for subset in subset_dataarray:
        subset = np.delete(subset, split, axis=1)
    return subset_dataarray


def fit(X_train, y_train):
    """
    Function implementing decision tree induction.
    
    :param X_train
        list of attribute values
    :param y_train
        target column aka the class
    :return
        trained decision tree (model)
    """
    attributes = len(X_train[0])
    data_array = np.concatenate((X_train, y_train[:,None]), axis = 1)
    classes_count = np.unique(y_train, return_counts=True)
    root = Node()

    # If all values of y_train is equal
    if stopCondition(classes_count) == True:
            for i in classes_count:
                # Also pretty shady, if there is time, fix
                if(i==0 or i==1):
                        root.setData(i)
                        return root

    # If there are no more attributes, but still more data in the array 
    # You must return the majority value of the classes 
    if attributes == 0:
        majority_value = np.argmax(classes_count[1])
        root.set_data(majority_value)
        return root

    split = bestSplit(X_train,y_train)
    attributes = attributes - 1

    for i in range(attributes):
        #Partitioning the data set given a unique attribute value 
        X_subset = partitionDataset(data_array,split)
        y_subset = []
        for sub in X_subset:
            y_subset.append(sub[:,-1])
        for i in range(len(X_subset)):
            child = fit(X_subset[i],y_subset[i])
            root.addChild(child)

    # Making a dictionary for the unique values of each attribute given columns  X_train
    
    
    # Use Hunt's algorithm
    
    # Find the best split
    return root

model = fit(X_train, y_train)
