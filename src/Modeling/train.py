import os
import mlflow
from mlflow import log_metric, log_param, log_artifact
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt


class Train:

    def __init__(self):
        pass

    def split_data(self, data, label):

        X = data.drop(columns=[label])
        y = data[label]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
        return X_train, X_test, y_train, y_test

    def model_xg(self, X_train, X_test, y_train, y_test, parameters):
        objective_choice = parameters["objective_choice"]
        class_weight = parameters["class_weight"]
        eval_metric_choice = parameters["eval_metric_choice"]

        clf = XGBClassifier(eta=0.3, objective=objective_choice, random_state=42, scale_pos_weight=class_weight,
                            max_depth=2, eval_metric=eval_metric_choice)

        clf.fit(X_train, y_train)
        y_test_pred = clf.predict(X_test)
        y_train_pred = clf.predict(X_train)

        return y_test_pred, y_train_pred, clf


    def model_rf(self, X_train, X_test, y_train, y_test, parameters):

        n_estimators = parameters["n_estimators"]
        class_weight = parameters["class_weight"]
        max_samples = parameters["max_samples"]
        random_state = parameters["random_state"]
        oob_score = parameters["oob_score"]
        min_samples_leaf = parameters["min_samples_leaf"]
        max_depth = parameters["max_depth"]

        clf = RandomForestClassifier(n_estimators = n_estimators, class_weight = class_weight, max_samples = max_samples,
                                     random_state = random_state, oob_score = oob_score, min_samples_leaf= min_samples_leaf,
                                     max_depth = max_depth)
        clf.fit(X_train, y_train)
        y_test_pred = clf.predict(X_test)
        y_train_pred = clf.predict(X_train)

        return y_test_pred, y_train_pred, clf

    def model_ada(self, X_train, X_test, y_train, y_test, parameters):
        pass

    def model_logistic(self,X_train, X_test, y_train, y_test, parameters):
        pass

    def training(self, data, label, parameters, model_name, experiment_name):

        X_train, X_test, y_train, y_test = self.split_data(data,label)

        mlflow.set_experiment(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)

        with mlflow.start_run():

            if model_name == "XG":
                y_test_pred, y_train_pred, model =  self.model_xg(X_train, X_test, y_train, y_test, parameters)

            if model_name == "RF":
                y_test_pred, y_train_pred, model = self.model_rf(X_train, X_test, y_train, y_test, parameters)

            from sklearn.metrics import average_precision_score
            average_precision = average_precision_score(y_test, y_test_pred)
            disp = plot_precision_recall_curve(model, X_test, y_test)
            disp.ax_.set_title('2-class Precision-Recall curve: '
                               'AP={0:0.2f}'.format(average_precision))
            plt.show()

            pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True)).transpose().to_csv("Test_report.csv",index=False)
            pd.DataFrame(classification_report(y_train, y_train_pred, output_dict=True)).transpose().to_csv("Train_report.csv",index=False)
            print("Test", classification_report(y_test, y_test_pred, output_dict=True))
            print("Train", classification_report(y_train, y_train_pred, output_dict=True))


            mlflow.log_artifact("Test_report.csv")
            mlflow.log_artifact("Train_report.csv")
            mlflow.sklearn.log_model(model,"model")








