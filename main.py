import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.utils import resample
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score

from matplotlib import pyplot


def down_sample(train, test):

    train_data = train.copy()

    # Separate majority and minority classes
    majority = train_data[train_data["Class"] == 0]
    minority = train_data[train_data["Class"] == 1]

    num_of_data = train_data["Class"].value_counts()[1]
    majority_downsampled = resample(
        majority, replace=False, n_samples=num_of_data, random_state=123
    )

    # Combine minority class with downsampled majority class
    df_downsampled = pd.concat([majority_downsampled, minority])

    print("Size of TRAIN is: " + str(df_downsampled.shape[0]))
    print(
        "Ratio of TRAIN is: "
        + str(df_downsampled["Class"].value_counts(normalize=True)[1])
    )

    X = df_downsampled.drop(columns=["Class"])
    y = df_downsampled["Class"]

    X_test = test.drop("Class", axis=1)
    y_test = test["Class"]
    print("Size of TEST is: " + str(test.shape[0]))
    print("Ratio of TEST is: " + str(test["Class"].value_counts(normalize=True)[1]))

    y_prob = run_random_forest(X, X_test, y)

    print(y_prob)

    return classification_metrics(y_test, y_prob)


def split_and_ensemble(train, test, random=True):
    # Get ratio of majority and minority
    ratio = train["Class"].value_counts()[0] / dataset["Class"].value_counts()[1]
    R = int(ratio)

    train_data = train.copy()

    X_test = test.drop("Class", axis=1)
    y_test = test["Class"]
    print("Size of TEST is: " + str(test.shape[0]))
    print("Ratio of TEST is: " + str(test["Class"].value_counts(normalize=True)[1]))

    # Separate majority and minority classes
    majority = train_data[train_data["Class"] == 0]
    minority = train_data[train_data["Class"] == 1]

    # Split into R groups
    if random:
        # split randomly
        split_majorities = np.array_split(majority, R)
    else:
        # split by clustering
        kmeans = KMeans(n_clusters=R)
        kmeans.fit(majority)
        predicted_majority = kmeans.predict(majority)

        majority["group"] = predicted_majority
        majority = majority.sort_values(by=["group"])

        majority = majority.drop(columns=["group"])
        split_majorities = np.array_split(majority, R)

    R_test_results = []
    i = 0
    for group in split_majorities:
        df = pd.concat([group, minority])

        X = df.drop(columns=["Class"])
        y = df["Class"]

        y_prob = run_random_forest(X, X_test, y)

        R_test_results.append(y_prob)

    y_prob_ensembled = np.mean(R_test_results, axis=0)

    return classification_metrics(y_test, y_prob_ensembled)


def run_random_forest(X_train, X_test, y_train):

    clf = RandomForestClassifier(n_estimators=20, random_state=0)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]

    return y_prob


def classification_metrics(y_test, y_prob):
    pred = y_prob.round()
    f1 = f1_score(y_test, pred)
    p = precision_score(y_test, pred)
    r = recall_score(y_test, pred)
    auc = roc_auc_score(y_test, y_prob)
    print("F1 Score: ", f1)
    print("Percision: ", p)
    print("Recall: ", r)
    print("AUC: ", auc)
    return [f1, p, r, auc]


if __name__ == "__main__":
    dataset = pd.read_csv("data/creditcard.csv")
    print(dataset.shape)
    rob_scaler = RobustScaler()

    # Standerize the data
    dataset["scaled_amount"] = rob_scaler.fit_transform(
        dataset["Amount"].values.reshape(-1, 1)
    )
    dataset["scaled_time"] = rob_scaler.fit_transform(
        dataset["Time"].values.reshape(-1, 1)
    )

    dataset.drop(["Time", "Amount"], axis=1, inplace=True)

    # Split to 10-Fold
    skfold = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
    fn = 0
    i = 0
    results_df = pd.DataFrame(
        columns=(
            "method",
            "fold_number",
            "class_ratio",
            "f1",
            "precision",
            "recall",
            "auc",
        )
    )

    for train_index, test_index in skfold.split(dataset, dataset["Class"]):
        print("**** Fold #", fn, " ****")
        print("Train:", train_index, "Test:", test_index)

        train, test = dataset.iloc[train_index], dataset.iloc[test_index]

        class_ratio = test["Class"].value_counts(normalize=True)[1]

        fold_results = down_sample(train, test)
        results_df.loc[i] = ["down_sample", fn, class_ratio] + fold_results

        fold_results_random = split_and_ensemble(train, test, True)
        results_df.loc[i + 1] = ["random_split", fn, class_ratio] + fold_results_random

        # TODO - make this work:
        fold_results_cluster = split_and_ensemble(train, test, False)
        results_df.loc[i + 2] = [
            "cluster_split",
            fn,
            class_ratio,
        ] + fold_results_cluster

        fn += 1
        i += 3

    results_df.to_csv("data.csv")
