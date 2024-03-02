import random

import pandas as pd
import numpy as np
import random
from geopy.geocoders import Nominatim

from .models import *

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import RegexpTokenizer
import os
import pickle
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score


import pandas as pd
import numpy as np
from collections import Counter
import pickle
import joblib


def load_trainingData_from_pickle():
    current_dir = os.getcwd()
    folder_name = "uploads"
    pickle_file_path = os.path.join(current_dir, folder_name, "train_data.pkl")
    if os.path.exists(pickle_file_path):
        try:
            x_train, y_train = joblib.load(pickle_file_path)
            return x_train, y_train
        except Exception as e:
            print("Error loading pickle file:", e)
            return None
    else:
        print("Pickle file does not exist.")
        return None


def LogisticRegression_Model():
    current_dir = os.getcwd()
    x_train, y_train = load_trainingData_from_pickle()
    x_train = x_train
    y_train = y_train
    ls = LogisticRegression()
    y_train = [0 if label == "good" else 1 for label in y_train]
    ls.fit(x_train, y_train)
    y_train_pred = ls.predict(x_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print("Training Accuracy:", train_accuracy)
    save_folder = "LogisticRegressionModel"
    save_filename = "Logisticmodeltrained.pkl"

    save_path = os.path.join(current_dir, save_folder, save_filename)

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    with open(save_path, "wb") as file:
        pickle.dump(ls, file)

    print("Model saved successfully at:", save_path)
    return True


def NaiveBayes_model():
    x_train, y_train = load_trainingData_from_pickle()
    print(x_train.shape)

    X_train = x_train
    y_train = y_train
    x_test = x_train
    y_test = y_train
    y_train = list(y_train)
    y_test = list(y_train)
    y_train = [0 if label == "good" else 1 for label in y_train]
    y_test = [0 if label == "good" else 1 for label in y_test]
    from imblearn.under_sampling import RandomUnderSampler

    random_sampler = RandomUnderSampler()
    x_train, y_train = random_sampler.fit_resample(X_train, y_train)

    print("Dimensions of the training features dataset:", x_train.shape)
    print(
        "No. of emails in the training features = no. of emails in the training labels?",
        len(y_train) == x_train.shape[0],
    )
    print("Dimensions of the test features dataset:", x_test.shape)
    print(
        "No. of emails in the test features = no. of emails in the test labels?",
        len(y_test) == x_test.shape[0],
    )

    print("Classes in the training dataset and their counts:", dict(Counter(y_train)))
    print("Classes in the testing dataset and their counts:", dict(Counter(y_test)))

    def calc_cond_probs(x_train, dirichlet_alpha):
        features_y0 = []
        features_y1 = []

        vocab_y0 = np.sum(x_train[:117684, :])
        vocab_y1 = np.sum(x_train[117684:, :])

        for i in range(x_train.shape[1]):
            temp_y0_feat = (np.sum(x_train[:117684, i]) + dirichlet_alpha) / (
                vocab_y0 + dirichlet_alpha * x_train.shape[1]
            )
            features_y0.append(temp_y0_feat)
            temp_y1_feat = (np.sum(x_train[117684:, i]) + dirichlet_alpha) / (
                vocab_y1 + dirichlet_alpha * x_train.shape[1]
            )
            features_y1.append(temp_y1_feat)

        return np.array(features_y0), np.array(features_y1)

    w_y0, w_y1 = calc_cond_probs(x_train, dirichlet_alpha=0.1)

    print("Feature weights for class 0:", w_y0[:5])
    print("Feature weights for class 1:", w_y1[:5])

    def calc_log_likelihood(f_y0, f_y1, x):
        log_likelihood_y0 = []
        log_likelihood_y1 = []

        for i in range(x.shape[0]):
            f_y0_arr = np.asarray(f_y0).flatten()
            f_y1_arr = np.asarray(f_y1).flatten()

            log_likelihood_y0.append(
                np.sum(np.log(f_y0_arr) * x[i, :].toarray().flatten())
            )
            log_likelihood_y1.append(
                np.sum(np.log(f_y1_arr) * x[i, :].toarray().flatten())
            )

        return log_likelihood_y0, log_likelihood_y1

    
    log_w_y0, log_w_y1 = calc_log_likelihood(w_y0, w_y1, x_train)

    print(
        "Log likelihoods of observations for class 0:",
        [round(number, 2) for number in log_w_y0[:5]],
    )
    print(
        "Log likelihoods of observations for class 1:",
        [round(number, 2) for number in log_w_y1[:5]],
    )

    def calc_posterior(logLH_y0, logLH_y1, y):
        epsilon = 1e-10

        log_prior_y0 = np.log(np.count_nonzero(y == 0) / len(y) + epsilon)
        log_prior_y1 = np.log(np.count_nonzero(y == 1) / len(y) + epsilon)

        posterior_y0 = logLH_y0 + log_prior_y0
        posterior_y1 = logLH_y1 + log_prior_y1

        return posterior_y0, posterior_y1

    posterior_y0, posterior_y1 = calc_posterior(log_w_y0, log_w_y1, y_train)

    print(
        "Posterior probabilities of observation for class 0:",
        [round(number, 2) for number in posterior_y0[:5]],
    )
    print(
        "Posterior probabilities of observation for class 1:",
        [round(number, 2) for number in posterior_y1[:5]],
    )

    def classify(p_y0, p_y1, y):

        class_pred = [
            0 if p_y0[i] > p_y1[i] or p_y0[i] == p_y1[i] else 1 for i in range(len(y))
        ]
        true_pos = len([i for i in range(len(y)) if y[i] == 0 and class_pred[i] == 0])
        true_neg = len([i for i in range(len(y)) if y[i] == 1 and class_pred[i] == 1])
        false_pos = len([i for i in range(len(y)) if y[i] == 0 and class_pred[i] == 1])
        false_neg = len([i for i in range(len(y)) if y[i] == 1 and class_pred[i] == 0])

        confusion_matrix = np.array([[true_pos, false_pos], [false_neg, true_neg]])
        accuracy = (true_pos + true_neg) / len(y)

        return class_pred, accuracy, confusion_matrix

    y_predicted, model_accuracy, model_confusion = classify(
        posterior_y0, posterior_y1, y_train
    )

    print("Model accuracy is {:0.2f}%".format(model_accuracy * 100))
    print("Training confusion matrix:")
    print(model_confusion)
    print("now running train_nb_modelpart")

    def train_NB_model(x_train, y_train, dirichlet_alpha, save_path):
        f_y0, f_y1 = calc_cond_probs(x_train, dirichlet_alpha)
        logLH_train_y0, logLH_train_y1 = calc_log_likelihood(f_y0, f_y1, x_train)
        posterior_train_y0, posterior_train_y1 = calc_posterior(
            logLH_train_y0, logLH_train_y1, y_train
        )
        train_class_pred, training_accuracy, train_confusion = classify(
            posterior_train_y0, posterior_train_y1, y_train
        )

        print(
            "Training classification accuracy: {:0.2f}%".format(training_accuracy * 100)
        )
        folder_path = os.path.dirname(save_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if os.path.exists(save_path):
            os.remove(save_path)

        with open(save_path, "wb") as file:
            pickle.dump((f_y0, f_y1), file)

        return f_y0, f_y1, train_class_pred, train_confusion

    save_folder = "NaiveBayesModel"
    save_filename = "NaiveModelTrained.pkl"
    current_dir = os.getcwd()

    save_path = os.path.join(current_dir, save_folder, save_filename)

    train_NB_model(x_train, y_train, 0.1, save_path)
    return True


def load_testingData_from_pickle():
    current_dir = os.getcwd()
    folder_name = "testfiles"
    pickle_file_path = os.path.join(current_dir, folder_name, "test_data.pkl")

    if os.path.exists(pickle_file_path):
        try:
            x_test, y_test = joblib.load(pickle_file_path)
            return x_test, y_test
        except Exception as e:
            print("Error loading pickle file:", e)
            return None
    else:
        print("Pickle file does not exist.")
        return None


def load_cv_pickle():
    current_dir = os.getcwd()
    folder_name = "files"
    pickle_file_path = os.path.join(current_dir, folder_name, "count_vectorizer.pkl")

    if os.path.exists(pickle_file_path):
        try:
            cv = joblib.load(pickle_file_path)
            return cv
        except Exception as e:
            print("Error loading pickle file:", e)
            return None
    else:
        print("Pickle file doesnot exist")


def load_NaivePickleModel():
    current_dir = os.getcwd()
    folder_name = "NaiveBayesModel"
    pickle_file_path = os.path.join(current_dir, folder_name, "NaiveModelTrained.pkl")

    if os.path.exists(pickle_file_path):
        try:
            with open(pickle_file_path, "rb") as file:
                f_y0, f_y1 = pickle.load(file)
            return f_y0, f_y1
        except Exception as e:
            print("Error loading pickle file:", e)
            return None
    else:
        print("Pickle file does not exist.")
        return None


def load_LogisticModel():
    current_dir = os.getcwd()
    folder_name = "LogisticRegressionModel"
    pickle_file_path = os.path.join(
        current_dir, folder_name, "Logisticmodeltrained.pkl"
    )

    if os.path.exists(pickle_file_path):
        try:
            ls = joblib.load(pickle_file_path)
            return ls
        except Exception as e:
            print("Error loading pickle file:", e)
            return None
    else:
        print("Pickle file does not exist.")
        return None


def NaiveBayes_testing():
    current_dir = os.getcwd()
    f_y0, f_y1 = load_NaivePickleModel()
    print(f_y0, f_y1)
    x_test, y_test = load_testingData_from_pickle()
    x_test = x_test
    y_test = y_test
    y_test = [0 if label == "good" else 1 for label in y_test]

    def calc_log_likelihood(f_y0, f_y1, x):
        log_likelihood_y0 = []
        log_likelihood_y1 = []

        for i in range(x.shape[0]):  
            f_y0_arr = np.asarray(f_y0).flatten()
            f_y1_arr = np.asarray(f_y1).flatten()

            log_likelihood_y0.append(
                np.sum(np.log(f_y0_arr) * x[i, :].toarray().flatten())
            )
            log_likelihood_y1.append(
                np.sum(np.log(f_y1_arr) * x[i, :].toarray().flatten())
            )

        return log_likelihood_y0, log_likelihood_y1

    log_w_y0, log_w_y1 = calc_log_likelihood(f_y0, f_y1, x_test)
    print("Feature weights for class 0:", log_w_y0[:5])
    print("Feature weights for class 1:", log_w_y1[:5])

    def calc_posterior(logLH_y0, logLH_y1, y):
        epsilon = 1e-10

        log_prior_y0 = np.log(np.count_nonzero(y == 0) / len(y) + epsilon)
        log_prior_y1 = np.log(np.count_nonzero(y == 1) / len(y) + epsilon)

        posterior_y0 = logLH_y0 + log_prior_y0
        posterior_y1 = logLH_y1 + log_prior_y1

        return posterior_y0, posterior_y1

    
    posterior_y0, posterior_y1 = calc_posterior(log_w_y0, log_w_y1, y_test)

    print(
        "Posterior probabilities of observation for class 0:",
        [round(number, 2) for number in posterior_y0[:5]],
    )
    print(
        "Posterior probabilities of observation for class 1:",
        [round(number, 2) for number in posterior_y1[:5]],
    )

    def classify(p_y0, p_y1, y):

       
        class_pred = [
            0 if p_y0[i] > p_y1[i] or p_y0[i] == p_y1[i] else 1 for i in range(len(y))
        ]

        true_pos = len([i for i in range(len(y)) if y[i] == 0 and class_pred[i] == 0])
        true_neg = len([i for i in range(len(y)) if y[i] == 1 and class_pred[i] == 1])
        false_pos = len([i for i in range(len(y)) if y[i] == 0 and class_pred[i] == 1])
        false_neg = len([i for i in range(len(y)) if y[i] == 1 and class_pred[i] == 0])

        confusion_matrix = np.array([[true_pos, false_pos], [false_neg, true_neg]])
        accuracy = (true_pos + true_neg) / len(y)

        return class_pred, accuracy, confusion_matrix

    y_predicted, model_accuracy, model_confusion = classify(
        posterior_y0, posterior_y1, y_test
    )
    print("now calling the testing function")

    def test_NB_model(x_test, y_test, f_y0, f_y1):
        
        logLH_test_y0, logLH_test_y1 = calc_log_likelihood(f_y0, f_y1, x_test)
        posterior_test_y0, posterior_test_y1 = calc_posterior(
            logLH_test_y0, logLH_test_y1, y_test
        )
        test_class_pred, test_accuracy, test_confusion = classify(
            posterior_test_y0, posterior_test_y1, y_test
        )

        print("Test classification accuracy: {:0.2f}%".format(test_accuracy * 100))

        return test_class_pred, test_confusion

    y_test_pred, test_confusion_matrix = test_NB_model(x_test, y_test, f_y0, f_y1)

    print("Test confusion matrix:")
    print(test_confusion_matrix)
    confusion_matrix = test_confusion_matrix
    TP = confusion_matrix[0, 0]
    FP = confusion_matrix[0, 1]
    FN = confusion_matrix[1, 0]
    TN = confusion_matrix[1, 1]

    confusion_matrix_instance = ConfusionMatrix.objects.create(
        true_positive=TP, false_positive=FP, false_negative=FN, true_negative=TN
    )
    confusion_matrix_instance.save()

    accuracy = accuracy_score(y_test, y_test_pred)
    precision_0 = precision_score(y_test, y_test_pred, pos_label=0)
    recall_0 = recall_score(y_test, y_test_pred, pos_label=0)
    f1_0 = f1_score(y_test, y_test_pred, pos_label=0)

    precision_1 = precision_score(y_test, y_test_pred, pos_label=1)
    recall_1 = recall_score(y_test, y_test_pred, pos_label=1)
    f1_1 = f1_score(y_test, y_test_pred, pos_label=1)

    print("Accuracy:", accuracy)
    print("Precision (Class 0):", precision_0)
    print("Recall (Class 0):", recall_0)
    print("F1-score (Class 0):", f1_0)
    print("Precision (Class 1):", precision_1)
    print("Recall (Class 1):", recall_1)
    print("F1-score (Class 1):", f1_1)

    metrics_instance = evaluationMetrics.objects.create(
        accuracy=accuracy,
        precision_class_0=precision_0,
        recall_class_0=recall_0,
        f1_class_0=f1_0,
        precision_class_1=precision_1,
        recall_class_1=recall_1,
        f1_class_1=f1_1,
    )
    metrics_instance.save()
    return True


def Logistic_testing():
    x_test, y_test = load_testingData_from_pickle()
    x_test = x_test[:50000]
    print(x_test.shape)
    y_test = y_test[:50000]
    x_test, y_test = load_testingData_from_pickle()
    y_test = [0 if label == "good" else 1 for label in y_test]
    ls = load_LogisticModel()
    y_pred = ls.predict(x_test)
    report = classification_report(y_pred, y_test)
    print(report)
    accuracy = accuracy_score(y_test, y_pred)
    precision_0 = precision_score(y_test, y_pred, pos_label=0)
    recall_0 = recall_score(y_test, y_pred, pos_label=0)
    f1_0 = f1_score(y_test, y_pred, pos_label=0)

   
    precision_1 = precision_score(y_test, y_pred, pos_label=1)
    recall_1 = recall_score(y_test, y_pred, pos_label=1)
    f1_1 = f1_score(y_test, y_pred, pos_label=1)
    conf_matrix = confusion_matrix(y_test, y_pred)

    TP = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]
    TN = conf_matrix[1, 1]
    print("Confusion Matrix:")
    print(conf_matrix)
    lr_confusion_matrix_instance = LogisticRegressionConfusionMatrix.objects.create(
        true_positive=TP, false_positive=FP, false_negative=FN, true_negative=TN
    )
    lr_confusion_matrix_instance.save()

   
    print("Accuracy:", accuracy)
    print("Precision (Class 0):", precision_0)
    print("Recall (Class 0):", recall_0)
    print("F1-score (Class 0):", f1_0)
    print("Precision (Class 1):", precision_1)
    print("Recall (Class 1):", recall_1)
    print("F1-score (Class 1):", f1_1)
    Logistic_metrics_instance = LogisticEvaluationReport.objects.create(
        accuracy=accuracy,
        precision_class_0=precision_0,
        recall_class_0=recall_0,
        f1_class_0=f1_0,
        precision_class_1=precision_1,
        recall_class_1=recall_1,
        f1_class_1=f1_1,
    )
    Logistic_metrics_instance.save()
    return True


def Single_url_check_Nb(Url):

    cv = load_cv_pickle()

    f_y0, f_y1 = load_NaivePickleModel()

    X_test_single_transformed = cv.transform([Url])
    print(X_test_single_transformed.shape)

    def calc_log_likelihood(f_y0, f_y1, x):
        log_likelihood_y0 = []
        log_likelihood_y1 = []

        for i in range(x.shape[0]):  
            f_y0_arr = np.asarray(f_y0).flatten()
            f_y1_arr = np.asarray(f_y1).flatten()

          
            log_likelihood_y0.append(
                np.sum(np.log(f_y0_arr) * x[i, :].toarray().flatten())
            )
            log_likelihood_y1.append(
                np.sum(np.log(f_y1_arr) * x[i, :].toarray().flatten())
            )

        return log_likelihood_y0, log_likelihood_y1

    def calc_posterior(logLH_y0, logLH_y1, y):
        epsilon = 1e-10

        log_prior_y0 = np.log(np.count_nonzero(y == 0) / len(y) + epsilon)
        log_prior_y1 = np.log(np.count_nonzero(y == 1) / len(y) + epsilon)

        posterior_y0 = logLH_y0 + log_prior_y0
        posterior_y1 = logLH_y1 + log_prior_y1

        return posterior_y0, posterior_y1

    log_likelihood_y0, log_likelihood_y1 = calc_log_likelihood(
        f_y0, f_y1, X_test_single_transformed
    )

    posterior_y0, posterior_y1 = calc_posterior(
        log_likelihood_y0, log_likelihood_y1, [0]
    )  

  
    if posterior_y0 > posterior_y1:
        predicted_label = 0 
    else:
        predicted_label = 1

    print("Predicted label by naive bayes", predicted_label)

    return predicted_label


def Single_url_check_lr(Url):
    cv = load_cv_pickle()
    ls = load_LogisticModel()
    X_test_single = [Url]
    X_test_single_transformed = cv.transform(X_test_single)
    print(X_test_single_transformed.shape)
    result = ls.predict(X_test_single_transformed)
    print("prediction by logistic regression", result)
    return result
