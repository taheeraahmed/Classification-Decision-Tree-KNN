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

    #Initializing the arrays we return
    X_train = np.zeros(shape=(len_train,22))    # 22 because of number of attributes
    y_train = np.zeros(shape=(len_train,22))    
    X_test = np.zeros(shape=(len_test,1))      # 1 because one class column
    y_test = np.zeros(shape=(len_test,1)) 

    # Merging the class and attribute columns
    data_array = np.concatenate((X, y[:,None]), axis = 1)
    # Shuffling the rows in data_array
    np.random.shuffle(data_array)

    # Split the training set 
    for row in data_array:
        # X_train is the topmost rows (len_train) in data_array, only the first 0-22 indices       
        print(row)
        # y_train is the topmost rows (len_train) in data_array, only the last index
        # X_test is the lower rows (len_test) in data_array, only the first 22 indces
        # y_test is the lower rows (len_test) in data_array, only the first last index
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, 0.33)