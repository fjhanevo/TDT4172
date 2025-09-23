import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linear_regression import LinearRegression
from logistic_regression import LogisticRegression, accuracy

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
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


def decrypt_stream(data):
    return np.ceil(data).astype(int)%2

def get_best_stream():
    CSV_DIR = "csv_files/"
    FILE_TRAIN = "mission3_train.csv"
    FILE_TEST = "mission3_test.csv"

    train = pd.read_csv(CSV_DIR+FILE_TRAIN)
    test = pd.read_csv(CSV_DIR+FILE_TEST)

    X_train_org = train.drop("target", axis=1)
    y_train = train["target"]

    X_test_org = test.drop("target", axis=1)
    y_test = test["target"]

    decryption_methods = {
        'truncate': lambda s: s.astype(int)%2,
        'floor': lambda s: np.floor(s).astype(int) % 2,
        'ceil': lambda s: np.ceil(s).astype(int) % 2,
        'round': lambda s: np.round(s).astype(int) % 2,
    }

    results = []

    for col in X_train_org.columns:
        for method, decrypt in decryption_methods.items():
            X_train_copy = X_train_org.copy()
            X_test_copy = X_test_org.copy()

            X_train_copy[col] = decrypt(X_train_copy[col])
            X_test_copy[col] = decrypt(X_test_copy[col])

            model = DecisionTreeClassifier(random_state=42)
            model.fit(X_train_copy, y_train)

            y_prob = model.predict_proba(X_test_copy)[:, 1]
            score = roc_auc_score(y_test, y_prob)

            results.append({
                'column': col,
                'method': method,
                'roc_auc': score
            })
    best_result = max(results, key=lambda x: x['roc_auc'])
    print(best_result)




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
    col_scores = {}
    print(train["data_stream_3"].head(10))
    print(decrypt_stream(train["data_stream_3"].head(10)))
    for col in X_train.columns:
        X_train_copy = X_train.copy()
        X_test_copy = X_test.copy()

        # decrypt the column
        X_train_copy[col] = decrypt_stream(X_train_copy[col])
        X_test_copy[col] = decrypt_stream(X_test_copy[col])

        # train a simple model to find hte correct data stream
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train_copy, y_train)

        # use predict_proba and roc_auc_score to find correct data stream
        y_prob = clf.predict_proba(X_test_copy)[:,1]
        score = roc_auc_score(y_test, y_prob)
        col_scores[col] = score
        print(f"- {col}: ROC AUC = {score:.4f}")
        
    # find the best column
    best_col = max(col_scores, key=col_scores.get)

    # print(f"Best data stream: {best_col} with score: {col_scores[best_col]:.4f}")
    #
    # # train on best data 
    # X_train[best_col] = decrypt_stream(X_train[best_col])
    # X_test[best_col] = decrypt_stream(X_test[best_col])
    #
    # # optimize params with gridsearchCV
    # params = {
    #     'criterion' : ['gini', 'entropy', 'log_loss'],
    #     'max_depth' : [2,4,6,7,8, None],
    #     'min_samples_leaf': [1,5,10,15],
    #     'min_samples_split' : [2,10,20]
    # }
    #
    # # find optimal parameters
    # grid_search = GridSearchCV(
    #     estimator=DecisionTreeClassifier(random_state=42),
    #     param_grid=params,
    #     cv=5,
    #     scoring='roc_auc',
    #     n_jobs=-1
    # )
    # grid_search.fit(X_train, y_train)
    # print(f"Best params: {grid_search.best_params_}")
    # best_clf = grid_search.best_estimator_
    # y_prob = best_clf.predict_proba(X_test)[:,1]
    #
    # ### ROC curve plot
    # fpr, tpr, _ = roc_curve(y_test, y_prob)
    # roc_auc = auc(fpr, tpr)
    # plt.plot(fpr, tpr, color="red", label=f"ROC curve (AUC = {roc_auc:.2f})")
    # plt.xlabel("FPR")
    # plt.ylabel("TPR")
    # plt.legend()
    # plt.show()

def main():
    # mission1()    
    # mission2()
    # mission3()
    get_best_stream()

if __name__ == '__main__':
    main()
