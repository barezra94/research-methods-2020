import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample
from sklearn import metrics


def split_data(path, label_name):
    dataset = pd.read_csv(path)

    X = dataset.drop(columns=[label_name])
    y = dataset[label_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    return X_train, X_test, y_train, y_test


def down_sample(path, label_name):
    dataset = pd.read_csv(path)

    # Separate majority and minority classes
    majority = dataset[dataset[label_name] == 0]
    minority = dataset[dataset[label_name] == 1]

    num_of_data = dataset[label_name].value_counts()[1]
    majority_downsampled = resample(
        majority, replace=False, n_samples=num_of_data, random_state=123
    )

    # Combine minority class with downsampled majority class
    df_downsampled = pd.concat([majority_downsampled, minority])

    X = df_downsampled.drop(columns=[label_name])
    y = df_downsampled[label_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    return X_train, X_test, y_train, y_test


def run_random_forest(X_train, X_test, y_train, y_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    regressor = RandomForestRegressor(n_estimators=20, random_state=0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
    print(
        "Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    )


if __name__ == "__main__":
    # X_train, X_test, y_train, y_test = split_data("data/creditcard.csv", "Class")
    # run_random_forest(X_train, X_test, y_train, y_test)

    down_sample("data/creditcard.csv", "Class")
