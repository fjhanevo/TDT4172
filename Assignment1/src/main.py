import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linear_regression import LinearRegression
from logistic_regression import LogisticRegression, accuracy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures

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


    # training 
    train_epochs = 30
    learning_rate = 0.1
    ### Output BEFORE scaling the data ### 

    lr0 = LogisticRegression(learning_rate=learning_rate, epochs=train_epochs)

    lr0.fit(X_train, y_train)
    y_pred0 = lr0.predict(X_test)
    print("Accuracy before scaling:", accuracy(y_test, y_pred0))
    ###

    ### Output AFTER scaling the data ### 
    mm_scaler = MinMaxScaler()
    X_train_scaled = mm_scaler.fit_transform(X_train)
    X_test_scaled = mm_scaler.transform(X_test)

    lr1 = LogisticRegression(learning_rate=learning_rate, epochs=train_epochs)
    lr1.fit(X_train_scaled, y_train)
    y_pred1 = lr1.predict(X_test_scaled)
    print("Accuracy after scaling:", accuracy(y_test, y_pred1))

    ### Output after Poly feat.
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    lr2 = LogisticRegression(learning_rate=learning_rate, epochs=train_epochs)

    lr2.fit(X_train_poly, y_train)
    y_pred2 = lr2.predict(X_test_poly)
    print("Accuracy after poly:", accuracy(y_test, y_pred2))

    """
    Conclusion: 
    Logisitic regression can not handle this dataset
    """


def main():
    # mission1()    
    mission2()

if __name__ == '__main__':
    main()
