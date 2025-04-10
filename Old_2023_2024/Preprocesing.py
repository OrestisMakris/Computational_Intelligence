import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class DataPreprocessor:
    def __init__(self, max_features=18000):
        self.max_features = max_features
        self.vectorizer = None
        self.scaler = None
        self.pipeline = None

    def read_csv(self, file_path):
        # Load data
        df = pd.read_csv(file_path, sep="\t")  # If the separator is tab
    
        # Drop specified columns
        columns_to_drop = ['id', 'metadata', 'region_main', 'region_sub', 'date_str', 'date_circa']
        df = df.drop(columns=columns_to_drop)
    
        return df

    def preprocess(self, texts, date_min, date_max, region_main_id, region_sub_id):
        # Combine text features
        combined_features = self.combine_features(texts, date_min, date_max, region_main_id, region_sub_id)
        # Perform cross-validation and return the folds
        folds = self.cross_validate(combined_features)
        # Print the first fold
        first_fold = folds[0]
        print("First Fold - Last 5 Columns:")
        print(first_fold[1][:, 150:])  # Print last 5 columns of the first fold
        # Return all folds
        return folds

    def combine_features(self, texts, date_min, date_max, region_main_id, region_sub_id):
        # Initialize and fit TfidfVectorizer
        self.vectorizer = self.create_tfidf_vectorizer(texts)
        X_text = self.vectorizer.transform(texts).toarray()
        
        # Calculate the midpoint date
        y_midpoint = (np.array(date_min) + np.array(date_max)) / 2
    
        # Convert date_min and date_max to numpy arrays
        X_date_min = np.array(date_min).reshape(-1, 1)
        X_date_max = np.array(date_max).reshape(-1, 1)
        X_region_main_id = np.array(region_main_id).reshape(-1, 1)
        X_region_sub_id = np.array(region_sub_id).reshape(-1, 1)
    
        # Concatenate the features 
        X = np.concatenate((X_date_min, X_date_max, X_region_main_id, X_region_sub_id), axis=1)
        
        # Initialize and fit MinMaxScaler
        self.scaler = self.create_minmax_scaler(X)
        # Normalize the features
        X_normalized = self.scaler.transform(X)

        # Combine TF-IDF features with scaled non-TF-IDF features
        X_normalized = np.concatenate((X_text, X_normalized), axis=1)

        return X_normalized, y_midpoint

    def create_tfidf_vectorizer(self, texts):
        vectorizer = TfidfVectorizer(max_features=self.max_features)
        vectorizer.fit(texts)
        return vectorizer

    def create_minmax_scaler(self, X):
        scaler = MinMaxScaler()
        scaler.fit(X)
        return scaler

    def cross_validate(self, X):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        folds = []
        for train_index, test_index in kf.split(X[0]):
            train_data = (X[0][train_index], X[1][train_index])
            test_data = (X[0][test_index], X[1][test_index])
            folds.append((train_data, test_data))
        return folds

class NNModel:
    def __init__(self, input_dim, hidden_nodes):
        self.input_dim = input_dim
        self.hidden_nodes = hidden_nodes
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            Dense(self.hidden_nodes, input_dim=self.input_dim, activation='relu'),
            Dense(1)
        ])
        return model

    def compile_model(self):
        self.model.compile(optimizer='adam', loss=self.rmse_loss)

    def rmse_loss(self, y_true, y_pred):
        # Custom RMSE calculation
        error = tf.abs(y_pred - y_true)
        min_error = tf.minimum(error, tf.abs(y_pred - tf.expand_dims(y_true[:, 0], axis=-1)))
        max_error = tf.minimum(error, tf.abs(y_pred - tf.expand_dims(y_true[:, 1], axis=-1)))
        return tf.sqrt(tf.reduce_mean(tf.square(tf.minimum(min_error, max_error))))

    def train(self, X_train, y_train, X_val, y_val, epochs=10):
        history = self.model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), verbose=1)
        return history

def main():
    dp = DataPreprocessor()
    df = dp.read_csv("./Data/iphi2802.csv")
    texts = df['text'].tolist()
    date_min = df['date_min'].tolist()
    date_max = df['date_max'].tolist()
    region_main_id = df['region_main_id'].tolist()
    region_sub_id = df['region_sub_id'].tolist()

    X, y_midpoint = dp.preprocess(texts, date_min, date_max, region_main_id, region_sub_id)
    
    for fold_idx, (train_data, test_data) in enumerate(X):
        X_train, y_train = train_data[0], y_midpoint[train_data[1]]
        X_test, y_test = test_data[0], y_midpoint[test_data[1]]

        # Define and compile the neural network model
        input_dim = X_train.shape[1]
        hidden_nodes = 10  # You can experiment with different numbers of hidden nodes
        nn_model = NNModel(input_dim=input_dim, hidden_nodes=hidden_nodes)
        nn_model.compile_model()

        # Train the neural network model
        history = nn_model.train(X_train, y_train, X_test, y_test, epochs=10)

        # Evaluate the model
        predictions = nn_model.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print(f'Fold {fold_idx + 1} RMSE: {rmse}')

if __name__ == "__main__":
    main()
