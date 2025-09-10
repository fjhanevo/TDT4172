import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linear_regression import LinearRegression
from logistic_regression import LogisticRegression, accuracy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier

def mission1():
    CSV_DIR = "csv_files/"
    FILE = "mission1.csv"

    data = pd.read_csv(CSV_DIR+FILE)

    # extract data
    X = data['Net_Activity'].values.reshape(-1,1)   # reshape for correct dimensions
    y = data['Energy'].values

    # linear regression
    lr = LinearRegression()
    lr.fit(X, y)
    y_pred = lr.predict(X)

    lr.plot_prediction_error_dist(X,y)

    plt.figure(figsize=(6,4))
    plt.scatter(X, y, c='blue', label='Data points')
    plt.plot([min(X), max(X)], [min(y_pred), max(y_pred)], c='red', label='Linear fit')
    plt.grid(True)
    plt.xlabel('Network Activity', fontsize=14)
    plt.ylabel('Energy', fontsize=14)
    plt.title('Energy vs. Traffic', fontsize=16)
    plt.legend()
    plt.show()

def mission2():
    CSV_DIR = "csv_files/"
    FILE = "mission2.csv"

    data = pd.read_csv(CSV_DIR+FILE)

    X_data = data[["x0", "x1"]]
    y_data = data["y"]

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)


    ### Training 
    train_epochs = 150
    learning_rate = 0.08

    ### Output after Poly feat.
    poly = PolynomialFeatures(degree=3)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    lr = LogisticRegression(learning_rate=learning_rate, epochs=train_epochs)

    lr.fit(X_train_poly, y_train)
    y_pred = lr.predict(X_test_poly)
    acc = accuracy(y_test, y_pred)
    print("Accuracy:", acc)

    ### ROC curve plot
    y_prob = lr.predict_proba(X_test_poly)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr, tpr, color="red", label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.show()


def decrypt_feat(series):
    return series.astype(int)%2

def mission3():
    CSV_DIR = "csv_files/"
    FILE_TRAIN = "mission3_train.csv"
    FILE_TEST = "mission3_test.csv"

    train = pd.read_csv(CSV_DIR+FILE_TRAIN)
    test = pd.read_csv(CSV_DIR+FILE_TEST)

    X_train = train.drop("target", axis=1)
    y_train = train["target"]

    X_test = test.drop("target", axis=1)
    y_test = test["target"]
    accuracies = {}
    for col in X_train.columns:
        X_train_copy = X_train.copy()
        X_test_copy = X_test.copy()

        # decrypt the column
        X_train_copy[col] = decrypt_feat(X_train_copy[col])
        X_test_copy[col] = decrypt_feat(X_test_copy[col])

        # train/test decision tree
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train_copy, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test_copy))
        accuracies[col] = acc

    # find the best column
    best_col = max(accuracies, key=accuracies.get)
    best_acc = accuracies[best_col]

    print("Correct datastream:", best_col, "with accuracy:", best_acc)

    # train on best data 
    X_train[best_col] = decrypt_feat(X_train[best_col])
    X_test[best_col] = decrypt_feat(X_test[best_col])

    clf = DecisionTreeClassifier(random_state=42, max_depth=3, min_samples_leaf=2, min_samples_split=5)
    clf.fit(X_train, y_train)

    ### ROC curve plot
    y_prob = clf.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color="red", label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.show()

def main():
    # mission1()    
    # mission2()
    mission3()

if __name__ == '__main__':
    main()
