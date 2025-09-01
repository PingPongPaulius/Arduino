import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import re
import os
import serial
import time

def read_file(path: str) -> pd.DataFrame:
    """
    Find csv file in directory
    """
    files: list[str] = os.listdir("data")
    for file in files:
        if ".csv" in file:
            return pd.read_csv(path + "/" + file)

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process Data to have inputs, and we only care about land weather.
    """
    column_names = list(df.columns)
    columns_to_drop = [column_names[i] for i in [2,3,4,5,6,8]]
    df = df.drop(columns_to_drop, axis=1)
    df = df[df['Location'] == 'inland']
    df = df.drop('Location', axis=1)

    max_humidity = max(df['Humidity'])
    df['Humidity'] = df['Humidity'] / float(max_humidity) * 100

    df['Season'] = df['Season'].map({'Autumn': 0, 'Winter': 1, 'Spring': 2, 'Summer': 3})

    print(len(df), len(df) * 0.2)
    
    return df

def main():
    
    arduino = serial.Serial('/dev/ttyACM0', baudrate=9600, timeout=1, dsrdtr=False)
    df: pd.DataFrame = read_file("data")
    df = process_data(df)

    output_column = 'Weather Type'
    X = df.drop(output_column, axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, df[output_column], test_size = 0.2, random_state=3)

    classifiers = {}
    classifiers['Decision Tree'] = tree.DecisionTreeClassifier()
    classifiers['Nearest Neighbours'] = KNeighborsClassifier()
    classifiers['Bayes'] = GaussianNB()
    classifiers['Quadratic Discriminant Analysis'] = QuadraticDiscriminantAnalysis()
    classifiers['Ada'] = AdaBoostClassifier()
    classifiers['Random Forest'] = RandomForestClassifier()
    classifiers['MLP'] = MLPClassifier()
    classifiers['Support Vectors'] = SVC()

    for name, clf in classifiers.items():

        clf = clf.fit(X_train, Y_train)
        acc = clf.score(X_test, Y_test)
        print(f"{name}: {acc}")

    while True:
        if arduino.in_waiting > 0:
            data = arduino.readline().decode('utf-8').strip()
            if data:
                print(data)
                matches = re.findall(r"[-+]?\d*\.\d+|\d+", data)
                temperature = float(matches[0])
                humidity = float(matches[1])
                season = 0

                real_data = pd.DataFrame({"Temperature": [temperature], "Humidity": [humidity], "Season": [season]})
                print(real_data)
                for name, clf in classifiers.items():
                    
                    acc = clf.predict(real_data)
                    print(f"{name}: {acc}")

            break


if __name__ == "__main__":
    main()
