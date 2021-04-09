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
    # This is how many rows we want our test set to be
    len_test = int(test_size * len(X))
    
    # This is how many rows we want our train set to be
    len_train = int((1-test_size) * len(X))+1

    # Merging the class and attribute columns
    data_array = np.concatenate((X, y[:,None]), axis = 1)
    # Shuffling the rows in data_array
    np.random.shuffle(data_array)
    
    # Slicing the data array such that it fits the return values
    X_train = data_array[:len_train,:-1]# shape = (len_train,22)
    y_train = data_array[:len_train,-1] # shape = (len_train,1)
    X_test = data_array[len_test:,:-1]  # shape = (len_train,22)
    y_test = data_array[len_test:,-1]   # shape = (len_train,1)
    
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, 0.33)