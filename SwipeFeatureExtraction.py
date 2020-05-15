import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.metrics import confusion_matrix, make_scorer
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle as pc

def extract_features(window_length):
    # iterate through each user
    for user_id in range(1, 197):
        print("Starting User" + str(user_id))

        # determine the type of data (horizontal phone, or vertical phone)
        if user_id < 139:
            session_type = "ps"
        else:
            session_type = "ls"

        # iterate through each session
        for session in ['1', '2']:
            features_by_window = []

            # load the data
            pickle_in = open("SwipeDataByUser\\User" + str(user_id) + "\\" + session_type + session, "rb")
            current_data = pc.load(pickle_in)
            pickle_in.close()

            # find windows
            windows = []
            i = 1
            while i + window_length <= len(current_data):
                windows.append(current_data[i:i + window_length])
                i += window_length // 2

            window_id = 0
            for window in windows:
                features_for_window = []
                for swipe in window:
                    features = []
                    # find the length of the swipe dataframe
                    len_swipe = len(swipe)
                    init_time = swipe.head(1)['EventTime']

                    # adds regression coef
                    if len_swipe > 1:
                        # do linear regression to find slope of swipe
                        reg = linear_model.LinearRegression(fit_intercept=False).fit(np.transpose([swipe["X"]]),
                                                                                     swipe["Y"])
                        features.append(reg.coef_[0])
                    else:
                        features.append(0)

                    # adds x_fin - x_init, y_fin - y_init, slope, distance, and time difference
                    x_init, y_init = swipe["X"][0], swipe["Y"][0]
                    x_fin, y_fin = swipe["X"][len_swipe - 1], swipe["Y"][len_swipe - 1]
                    t_init, t_fin = swipe["EventTime"][0], swipe["EventTime"][len_swipe - 1]

                    features.append(x_fin - x_init)
                    features.append(y_fin - y_init)
                    features.append((x_fin - x_init)**2 + (y_fin - y_init)**2)
                    features.append(t_fin - t_init)

                    # adds arclength
                    arclength = 0
                    for num in range(0, len_swipe - 1):
                        x_init, y_init = swipe["X"][num], swipe["Y"][num]
                        x_fin, y_fin = swipe["X"][num + 1], swipe["Y"][num + 1]
                        dist = (x_fin - x_init)**2 + (y_fin - y_init)**2
                        arclength += dist
                    features.append(arclength)

                    # second features
                    speeds = []
                    pressures = []
                    areas = []
                    for num in range(0, len_swipe - 1):
                        distance = (swipe["X"][num] - swipe["X"][num + 1]) ** 2 + (
                                swipe["Y"][num] - swipe["Y"][num + 1]) ** 2
                        current_speed = distance / (swipe["EventTime"][num + 1] - swipe["EventTime"][num])
                        speeds.append(current_speed)

                        pressures.append(swipe["Pressure"][num])

                        areas.append(swipe["Area"][num])

                    for lst in [speeds, pressures, areas]:
                        if len(lst) == 0:
                            mean = 0
                            std = 0
                            Q1 = 0
                            median = 0
                            Q3 = 0

                        else:
                            mean = np.mean(lst)
                            std = np.std(lst)
                            Q1 = np.percentile(lst, 25)
                            median = np.percentile(lst, 50)
                            Q3 = np.percentile(lst, 75)

                        features.append(mean)
                        features.append(std)
                        features.append(Q1)
                        features.append(median)
                        features.append(Q3)

                    if len(features_by_window) == 0:
                        for num in range(0,len(features)):
                            features_by_window.append([features[num]])

                    else:
                        for num in range(0,len(features)):
                            a = features_by_window[num]
                            features_by_window[num].append(features[num])

                features_by_window_avg = []
                for num in range(0, len(features_by_window)):
                    features_by_window_avg.append(np.mean(features_by_window[num]))
                # make new path if one doesn't exist
                path = "FeatureByUserAndWindow\\User" + str(user_id) + str(window_id)
                if not os.path.exists(path):
                    os.makedirs(path)

                pickle_out = open(path + "\\" 'session' + session, "wb")
                pc.dump(features_by_window_avg, pickle_out)
                pickle_out.close()
                window_id += 1
                print(features_by_window_avg)


def get_error_rates(training_x, training_y, gen_test_x, imp_test_x, classification_method):
    if classification_method == "kNN":  # This is an example of how can you use kNN
        n_neighbors = [int(x) for x in range(5, 10, 1)]
        # print('n_neighbors',n_neighbors)
        dist_met = ['manhattan', 'euclidean']
        # create the random grid
        param_grid = {'n_neighbors': n_neighbors,
                      'metric': dist_met}
        CUAuthModel = KNeighborsClassifier()
        scoring_function = 'f1'  # You can use scoring function as HTER, see Aux_codes.py for details
        # Grid search for best parameter search .. using 10 fold cross validation and 'f1' as a scoring function.
        SearchTheBestParam = GridSearchCV(estimator=CUAuthModel, param_grid=param_grid, cv=10,
                                          scoring=scoring_function)
        SearchTheBestParam.fit(training_x, training_y)
        best_nn = SearchTheBestParam.best_params_['n_neighbors']
        best_dist = SearchTheBestParam.best_params_['metric']

        # Retraining the model again using the best parameter and testing, remember k = 1 will always give 100% accuracy :) on training data
        FinalModel = KNeighborsClassifier(n_neighbors=best_nn, metric=best_dist)
        FinalModel.fit(training_x, training_y)

        pred_gen_lables = FinalModel.predict(gen_test)
        pred_imp_lables = FinalModel.predict(imp_test)
        pred_testing_labels = np.concatenate((pred_gen_lables, pred_imp_lables))

        testing_y = []
        for i in range(0, len(gen_test)):
            testing_y.append(1)
        for i in range(0, len(imp_test)):
            testing_y.append(0)

        # computing the error rates for the current predictions
        tn, fp, fn, tp = confusion_matrix(testing_y, pred_testing_labels).ravel()



        # far = fp / (fp + tn)
        # frr = fn / (fn + tp)
        # hter = (far + frr) / 2
        return tn, fp, fn, tp

    elif classification_method == "LogReg":  # This is an example of how can you use Logistic regression
        param_grid = [
            {'solver': ['newton-cg'],
             'C': [0.1, 0.2, 0.4, 0.45, 0.5],
             'penalty': ['l1', 'l2']}]  # Trying to improve

        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        CUAuthModel = linear_model.LogisticRegression(random_state=random_seed, tol=1e-5)
        scoring_function = 'f1'  # You can use scoring function as HTER, see Aux_codes.py for details
        # Grid search for best parameter search .. using 10 fold cross validation and 'f1' as a scoring function.
        SearchTheBestParam = GridSearchCV(estimator=CUAuthModel, param_grid=param_grid, cv=10,
                                          scoring=scoring_function)
        SearchTheBestParam.fit(training_x, training_y)
        solver = SearchTheBestParam.best_params_['solver']
        cval = SearchTheBestParam.best_params_['C']
        penalty = SearchTheBestParam.best_params_['penalty']

        # Retraining the model again using the best parameter and testing, remember k = 1 will always give 100% accuracy :) on training data
        FinalModel = linear_model.LogisticRegression(solver=solver, C=cval, penalty=penalty,
                                                     random_state=random_seed, tol=1e-5)
        FinalModel.fit(training_x, training_y)
        pred_gen_lables = FinalModel.predict(gen_test)
        pred_imp_lables = FinalModel.predict(imp_test)
        pred_testing_labels = np.concatenate((pred_gen_lables, pred_imp_lables))

        testing_y = []
        for i in range(0, len(gen_test)):
            testing_y.append(1)
        for i in range(0, len(imp_test)):
            testing_y.append(0)

        # computing the error rates for the current predictions
        tn, fp, fn, tp = confusion_matrix(testing_y, pred_testing_labels).ravel()
        # far = fp / (fp + tn)
        # frr = fn / (fn + tp)
        # hter = (far + frr) / 2
        return tn, fp, fn, tp

    else:  # Add more classification methods same as above
        raise ValueError('classification method unknown!')

if __name__ == "__main__":
    extract_features(100)






