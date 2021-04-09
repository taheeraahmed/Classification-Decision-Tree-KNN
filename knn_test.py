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
    distances = []
    for row_train in data_array:
        # Calculating the euclidean distance between the row in the test set and the row in the train set. 
        dist = euclideanDistance(row_train,row_test)
        distances.append((row_train, dist))
    # Sorting the distances list with respect to the distance in the tuple
    distances.sort(key=lambda tup: tup[1])

def getPrediction():
    return 0


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
    ### START CODE HERE ### 

    distances = np.zeros(shape=(len(X_train),1))

    # Sort all of the records in the training dataset by their distance to the new data.
    for row_test in X_test:
        neighbors = getNeighbors(row_test, X_test, k)

    
    ### END CODE HERE ### 
    return 0    #y_pred

y_hat = knn(X_train, y_train, X_test, k=5)