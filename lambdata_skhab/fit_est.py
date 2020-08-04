"""
Get a dataframe and print some statistics about it.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


class df_fit:
    def __init__(self, X):
        self.X = X

    def unique_df(self):
        """ Reports a coiunt of unique values in each column """

        nunique_sort = self.X.nunique().sort_values(ascending=False)
        # data_types not being used
        # data_types = self.X.dtypes
        return nunique_sort

    def skew_col(self, imb=0.99):
        """
        Returns the sorted list of feature names
        with imbalance amount exceeding "imb" parameter
        """

        self.X = self.X.copy()
        # self.X.nunique().index is the same thing as self.X.columns but longer to type
        # Syntax is incorrect. It should be pd.Series(
        # data=(
        # self.X[col].value_counts().max() / self.X[col].value_counts().sum()
        # for col in self.X.columns)
        # )
        # get the skew value of each column
        mask = pd.Series(
            data=(
                self.X[col].value_counts().max() / self.X[col].value_counts().sum()
                for col in self.X.columns
            )
        ).sort_values(ascending=False)

        skew_series = mask[mask >= imb]

        return skew_series

    # forget to add in self
    def fit_rep(self, estimator, X_train, y_train, X_val, y_val):
        """
        fit a pipeline estimator and return the estimator, y_val predictions, and score values
        """

        print("\n fitting ...")
        estimator.fit(X_train, y_train)

        print("\n predicting ...")
        y_pred = estimator.predict(X_val)

        score_train = estimator.score(X_train, y_train)
        score_val = estimator.score(X_val, y_val)

        print("\n score ...")
        print("Training score", score_train)
        print("Validation score", score_val)

        accuracy_score_train = accuracy_score(y_train, estimator.predict(X_train))
        accuracy_score_val = accuracy_score(y_val, y_pred)

        print("\n Accuracy ...")
        print("Training Accuracy", accuracy_score_train)
        print("Validation Accuracy", accuracy_score_val)

        return estimator, y_pred, score_train, score_val

    # forget to add in self
    def metric_rep(self, estimator, X_val, y_val):
        """ Calculates accuracy, recal and precision of a classifier and plots the confusion matrix"""

        print("\n predicting y ...")
        y_pred = estimator.predict(X_val)

        print("\n plotting confusion matrix ...")
        plt.rcParams["figure.dpi"] = 80
        plot_confusion_matrix(
            estimator, X_val, y_val, values_format=".0f", xticks_rotation="vertical"
        )

        print("\n calculating confusion matrix ...")
        C = pd.DataFrame(confusion_matrix(y_val, y_pred))
        print(C)

        truth_sum = C.sum(axis=1)
        predict_sum = C.sum(axis=0)
        pred_t = pd.Series([C.iloc[i, i] for i in range(len(C))])

        recall = pred_t / truth_sum
        precision = pred_t / predict_sum
        accuracy = pred_t.sum() / truth_sum.sum()

        print("\n classification report ...")
        print(
            classification_report(
                y_val, y_pred, target_names=estimator.classes_.astype(str)
            )
        )

        print(f"******\n accuracy is {accuracy:.2f}\n******")

        return accuracy, precision, recall
