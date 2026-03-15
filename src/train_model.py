# train_model.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

import config


def load_data():
    data = pd.read_csv(config.DATA_PATH)
    return data


def preprocess_data(data):
    X = data.drop(config.TARGET_COLUMN, axis=1)
    y = data[config.TARGET_COLUMN]
    return X, y


def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(),
        "xgboost": XGBClassifier(eval_metric="logloss")
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


def evaluate_models(models, X_test, y_test):

    results = {}

    for name, model in models.items():

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        print("\nModel:", name)
        print("Accuracy:", accuracy)
        print(classification_report(y_test, predictions))

        results[name] = accuracy

    return results


def save_best_model(models, results):

    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]

    print("\nBest Model:", best_model_name)

    joblib.dump(best_model, config.MODEL_PATH)

    print("Model saved at:", config.MODEL_PATH)


def main():

    data = load_data()

    X, y = preprocess_data(data)

    X_train, X_test, y_train, y_test = split_data(X, y)

    models = train_models(X_train, y_train)

    results = evaluate_models(models, X_test, y_test)

    save_best_model(models, results)


if __name__ == "__main__":
    main()