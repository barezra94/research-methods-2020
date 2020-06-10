import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.utils import resample
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score

from matplotlib import pyplot


def down_sample(X_train, X_test, y_train, y_test):

    dataset = X_train.copy()
    dataset["Class"] = y_train.values

    # Separate majority and minority classes
    majority = dataset[dataset["Class"] == 0]
    minority = dataset[dataset["Class"] == 1]

    num_of_data = dataset["Class"].value_counts()[1]
    majority_downsampled = resample(
        majority, replace=False, n_samples=num_of_data, random_state=123
    )

    # Combine minority class with downsampled majority class
    df_downsampled = pd.concat([majority_downsampled, minority])

    X = df_downsampled.drop(columns=["Class"])
    y = df_downsampled["Class"]

    y_pred = run_random_forest(X, X_test, y)

    classification_metrics(y_test, y_pred)


def random_split(dataset, label_name):
    # Get ratio of majority and minority
    ratio = (
        dataset[label_name].value_counts()[0] / dataset[label_name].value_counts()[1]
    )

    # Separate majority and minority classes
    majority = dataset[dataset[label_name] == 0]
    minority = dataset[dataset[label_name] == 1]

    # Split into N (=ratio) groups
    split_majorities = np.array_split(majority, ratio)

    # Run N classification models on split_majorities
    for group in split_majorities:
        df = pd.concat([group, minority])

        X = df.drop(columns=[label_name])
        y = df[label_name]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )

        # y_pred = run_random_forest(X_train, X_test, y_train, y_test)

        # TODO: Ensamble the results


def cluster_split(dataset, label_name):
    # Get ratio of majority and minority
    ratio = (
        dataset[label_name].value_counts()[0] / dataset[label_name].value_counts()[1]
    )

    ratio = int(ratio)

    # Separate majority and minority classes
    majority = dataset[dataset[label_name] == 0]
    minority = dataset[dataset[label_name] == 1]


def run_random_forest(X_train, X_test, y_train):

    regressor = RandomForestRegressor(n_estimators=20, random_state=0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    return y_pred


def classification_metrics(y_test, y_pred):
    copy_pred = y_pred.round()
    print("F1 Score: ", f1_score(y_test, copy_pred))
    print("Percision: ", precision_score(y_test, copy_pred))
    print("Recall: ", recall_score(y_test, copy_pred))
    print("AUC: ", roc_auc_score(y_test, copy_pred))


if __name__ == "__main__":
    dataset = pd.read_csv("data/creditcard.csv")
    rob_scaler = RobustScaler()

    # Standerize the data
    dataset["scaled_amount"] = rob_scaler.fit_transform(
        dataset["Amount"].values.reshape(-1, 1)
    )
    dataset["scaled_time"] = rob_scaler.fit_transform(
        dataset["Time"].values.reshape(-1, 1)
    )

    dataset.drop(["Time", "Amount"], axis=1, inplace=True)

    X = dataset.drop("Class", axis=1)
    y = dataset["Class"]

    # Split to 10-Fold
    sss = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
    n = 1

    for train_index, test_index in sss.split(X, y):
        print("**** Fold #", n, " ****")
        print("Train:", train_index, "Test:", test_index)
        Xtrain, Xtest = X.iloc[train_index], X.iloc[test_index]
        ytrain, ytest = y.iloc[train_index], y.iloc[test_index]

        down_sample(Xtrain, Xtest, ytrain, ytest)

        n += 1
