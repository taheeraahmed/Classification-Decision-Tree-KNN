import pandas as pd
import numpy as np

mushrooms_df = pd.read_csv('mushrooms.csv')

mushrooms_df

mushrooms_df.describe().T

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
y_df = mushrooms_encoded_df['class']  # classes
X_array = X_df.to_numpy()
y_array = y_df.to_numpy()

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
        a numpy array of train-test splits (X-train, X-test, y-train, y-test)
    """
    
    # This is how many rows we want our train set to be
    len_train = int((1-test_size) * len(X))

    # Merging the class and attribute columns
    data_array = np.concatenate((X, y[:,None]), axis = 1)
    # Shuffling the rows in data_array
    np.random.shuffle(data_array)
    
    # Slicing the data array such that it fits the return values
    X_train = data_array[:len_train,:-1]
    y_train = data_array[:len_train,-1]
    X_test = data_array[len_train:,:-1] 
    y_test = data_array[len_train:,-1] 
    
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, 0.33)

# Use this section to place any "helper" code for the `knn()` function.

def euclideanDistance(row1, row2):
	return np.linalg.norm(row1-row2)

def getNeighbors(row_test, data_array, k):
    """
    Getting the k nearest neighbor of a row in the test set given the training set
    
    :param row_test
        a row in the test set
    :param data_array
        all the rows in the training set
    :param k
        number of neighbors to use
    :return
        list of neighbors indexes in data_array
    """
    distances = []
    count = 0
    for row_train in data_array:
        # Calculating the euclidean distance between the row in the test set and the row in the train set. 
        dist = euclideanDistance(row_train,row_test)
        distances.append((count, dist))
        count += 1
    # Sorting the distances list with respect to the distance in the tuple
    distances.sort(key=lambda tup: tup[1])

    neighbors = []
    for neighbor in range(k):
        neighbors.append(distances[neighbor+1][0])
    return neighbors

def mostFrequent(list):
    return max(set(list), key = list.count)

def getPrediction(neighbors,classes):
    """
    Get the class/prediction of the neighbors and return the class which 
    belongs to most of the neighbors
    
    :param neighbors
        list of neighbors indexes in data_array
    :param classes
        all the rows in the training set
    :return
        class for row in test_set
    """
    predictions = []
    for neighbor in neighbors:
        # Finding row in the data array which is in neighbor 
        # Take the last column bc this is the class
        predictions.append(classes[neighbor,-1])
    most_frequent = mostFrequent(predictions)
    return mostFrequent(predictions)


def knn(X_train, y_train, X_test, k=5):
    """
    k-nearest neighbors classifier.
    
    :param X_train
        attributes of the groung truth (training set)
    :param y_train
        classes of the groung truth (training set)
    :param X_test
        attributes of samples to be classified
    :param k
        number of neighbors to use
    :return
        predicted classes
    """
    classes = np.concatenate((X_train, y_train[:,None]), axis = 1)
    predictions = []
    for row_test in X_test:
        neighbors = getNeighbors(row_test, X_train, k)
        predictions.append(getPrediction(neighbors, classes))

    return predictions

y_hat = knn(X_train, y_train, X_test, k=5)

def evaluate(y_true, y_pred):
    """
    Function calculating the accuracy of the model on the given data.
    
    :param y_true
        true classes
    :paaram y
        predicted classes
    :return
        accuracy
    """
    ### START CODE HERE ### 
    result = []
    for classes in y_true:
        print(classes)
    ### END CODE HERE ### 
    return accuracy

accuracy = evaluate(y_test, y_hat)
print('accuracy =', accuracy)