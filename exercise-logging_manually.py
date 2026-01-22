import argparse

import mlflow
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_neighbors", type=int, default=3)
args = vars(parser.parse_args())

# load in the data
df = pd.read_csv("data/winequality-white.csv", sep=";")
# if you want to you can combine the two datasets into one - but this is not necessary
df_red = pd.read_csv("data/winequality-red.csv", sep=";")
df_white = pd.read_csv("data/winequality-white.csv", sep=";")
df = pd.concat([df_red, df_white])

# check a few rows of the data - hint: use .head()
# print(df.head())

# Check for missing values
# print(df.isnull().sum())

# Do basic statistics of dataset
# print(df.describe().T)

# Splitting data
X = df.drop(columns="quality")
y = df["quality"]

# now split the data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

mlflow.set_experiment("wine_quality")

with mlflow.start_run():
    # Log the value of n_neighbors
    mlflow.log_param("n_neighbors", args["n_neighbors"])

    # create your pipeline
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=args["n_neighbors"])),
        ]
    )

    # fit the pipeline
    pipe.fit(X_train, y_train)

    # evaluate the pipeline
    y_pred = pipe.predict(X_test)

    # Log the train and test accuracy
    train_acc = accuracy_score(y_train, pipe.predict(X_train))
    test_acc = accuracy_score(y_test, pipe.predict(X_test))
    print("Train accuracy: ", train_acc)
    print("Test accuracy: ", test_acc)

    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)


# Check results of your runs by running 'mlflow ui' in your terminal
# Open the browser and go to http://localhost:5000 to see the resulting runs
