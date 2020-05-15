# Install packages as per the need
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

random_seed = 1833


# preprocess

# 1) Nearest line for swipe (Lin regression)
# starting point
# End point
# Average pressure (maybe just through middle fifty percent)
# Pressure over time
# max area pressed
# Second half speed

# classification

# result analysis


# This is a starter code for lab 5
# You are free to use this or develop your own from scratch
# Major components: pre-processing, feature extraction, feature analysis, classification, and performance evaluation

# Following two components are one-time and you don't want to do it again and again, unless you are trying many different
# methods of pre-processing and feature extractions
#################### Preprocessing ##################
def preprocess(data_path, path_type):
    data_frame = pd.read_csv(data_path)

    group = data_frame["UserID"]
    agg = data_frame.groupby([group])
    for user in agg:
        # make new path if one doesn't already exist
        user_swipes = []
        group = user[1]["SwipeID"]
        agg1 = user[1].groupby([group])
        for swipe in agg1:
            swipe[1].index = range(len(swipe[1].index))
            user_swipes.append(swipe[1])

        user_id = user[0]
        path = "DataByUser\\User" + str(user_id)
        if not os.path.exists(path):
            os.makedirs(path)

        # serialize all the data for each user
        pickle_out = open(path + "\\" + path_type, "wb")
        pc.dump(user_swipes, pickle_out)
        pickle_out.close()

        print(user_swipes)
    pass


###################  Feature extraction #############
# Extract all the features from all the users and save it in a folder using the following folder structure
# ExtractedFeatures\<User>\<Phone-hold-style>\<Session1/Session2> or whatever is suitable
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
            pickle_in = open("DataByUser\\User" + str(user_id) + "\\" + session_type + session, "rb")
            current_data = pc.load(pickle_in)
            pickle_in.close()

            # find windows
            windows = []
            i = 1
            while i + window_length <= len(current_data):
                windows.append(current_data[i:i + window_length])
                i += window_length // 2

            # iterate over each window
            window_id = 0
            for window in windows:
                # initialize the dictionary of features
                features = ([], [], [], [], [], [], [], [])

                # iterate over each swipe in the window
                for swipe in window:
                    # find the length of the swipe dataframe
                    len_swipe = len(swipe)
                    init_time = swipe.head(1)['EventTime']

                    if len_swipe > 1:
                        # do linear regression to find slope of swipe
                        reg = linear_model.LinearRegression(fit_intercept=False).fit(np.transpose([swipe["X"]]),
                                                                                     swipe["Y"])
                        features[0].append(reg.coef_[0])

                    # find starting and ending points of swipe
                    x_init, y_init = swipe["X"][0], swipe["Y"][0]
                    x_fin, y_fin = swipe["X"][len_swipe - 1], swipe["Y"][len_swipe - 1]
                    features[1].append(x_init)
                    features[2].append(y_init)

                    # max area of swipe
                    max_area = max(swipe["Area"])
                    features[3].append(max_area)

                    # get average pressure over time
                    avg_pressure = np.nanmean(swipe["Pressure"][len_swipe // 4:3 * len_swipe // 4 + 1])
                    features[4].append(avg_pressure)

                    if len_swipe > 1:
                        # get slope and intercept of pressure
                        reg = linear_model.LinearRegression().fit(np.transpose([swipe["EventTime"].values.tolist()]),
                                                                  swipe["Pressure"])
                        features[5].append(reg.coef_[0])
                        features[6].append(reg.intercept_)

                    # second half speed
                    speeds = []
                    for num in range(len_swipe // 2, len_swipe - 1):
                        distance = (swipe["X"][num] - swipe["X"][num + 1]) ** 2 + (
                                swipe["Y"][num] - swipe["Y"][num + 1]) ** 2
                        current_speed = distance / (swipe["EventTime"][num + 1] - swipe["EventTime"][num])
                        speeds.append(current_speed)
                    if len(speeds) == 0:
                        speed = 0
                    else:
                        speed = np.average(speeds)
                    features[7].append(speed)

                # average the list of features
                new_features = []
                for feature in features:
                    if len(feature) == 0:
                        new_features.append(0)
                    else:
                        new_feature = np.nanmean(feature)
                        new_features.append(new_feature)
                features_by_window.append(new_features)

                print(new_features)
            # make new path if one doesn't exist
            path = "FeatureByUserAndWindow\\User" + str(user_id)
            if not os.path.exists(path):
                os.makedirs(path)

            pickle_out = open(path + "\\" 'session' + session, "wb")
            pc.dump(features_by_window, pickle_out)
            pickle_out.close()


# finds session type, gen_train, imp_train, gen_test, imp_test
def get_user_model(user_id):
    # determine the type of data (horizontal phone, or vertical phone)
    if user_id < 139:
        session_type = "portrait"
        start = 1
        end = 139
    else:
        session_type = "landscape"
        start = 139
        end = 197

    file1 = "FeatureByUserAndWindow\\User" + str(user_id) + "\\session1"
    file2 = "FeatureByUserAndWindow\\User" + str(user_id) + "\\session2"
    if os.path.exists(file1) and os.path.exists(file2):
        gen_train_file = open(file1, "rb")
        gen_test_file = open(file2, "rb")
        gen_train_x = pc.load(gen_train_file)
        gen_train_file.close()
        gen_test = pc.load(gen_test_file)
        gen_test_file.close()

        if len(gen_train_x) > 0 and len(gen_test) > 0:
            imp_train_x = []
            imp_test = []
            for impostor in range(start, end):
                file1 = "FeatureByUserAndWindow\\User" + str(impostor) + "\\session1"
                file2 = "FeatureByUserAndWindow\\User" + str(impostor) + "\\session2"
                if impostor != user_id and os.path.exists(file1) and os.path.exists(file2):
                    imp_train_file = open(file1, "rb")
                    imp_test_file = open(file2, "rb")
                    imp_train_x = imp_train_x + pc.load(imp_train_file)
                    imp_train_file.close()
                    imp_test = imp_test + pc.load(imp_test_file)
                    imp_test_file.close()

            if len(imp_train_x) > 0 and len(imp_test) > 0:
                return session_type, gen_train_x, imp_train_x, gen_test, imp_test


def select_features(training_X, training_y, genuine_testing, impostor_testing, thresholdK):
    fselector = SelectKBest(mutual_info_classif, k=int(thresholdK))
    # Selecting k features based on mutual information
    # You can try other methods if you want to. Probably selecting features using mRmR will make more sense.
    fselector.fit(training_X, training_y)
    training_X = fselector.transform(training_X)
    genuine_testing = fselector.transform(genuine_testing)
    impostor_testing = fselector.transform(impostor_testing)
    return training_X, training_y, genuine_testing, impostor_testing  # returning the matrix with selected features


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


def analyse_results(far, frr, tar, trr):
    # plots the confusion matrix
    cm = [[tar, far], [frr, trr]]
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''])
    ax.set_yticklabels([''])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    pass


def scoringHTER(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    far = fp / (fp + tn)
    frr = fn / (fn + tp)
    hter = (far + frr) / 2
    if hter < 0:
        raise ValueError('ERROR: HTER CANT BE NEATIVE')
    return hter


# How to make scoring function for GridSearchCV
scorerHTER = make_scorer(scoringHTER, greater_is_better=False)


# Generating label column for the given data matrix
def get_labels(data_matrix, label):
    if data_matrix.shape[0] > 1:
        label_column = np.empty(data_matrix.shape[0])
        label_column.fill(label)
    else:
        print('Warning! user data contains only one sample')
    return label_column


# if __name__ == "__main__":
#     # this is the code to preprocess each type of data for each user
#     # preprocess("LandscapeSession1.csv", "ls1")
#     # preprocess("LandscapeSession2.csv", "ls2")
#     # preprocess("PortraitSession1.csv", "ps1")
#     # preprocess("PortraitSession2.csv", "ps2")
#
#     extract_features(10)

if __name__ == "__main__":
    final_result = []
    row_counter = 0
    classification_methods = ['kNN', 'LogReg']
    phone_usage_mode = ['landacape', 'portrait']
    user_list = []  # Get the user list
    feature_selection_threshold = 7  # Try different numbers of features
    SMOTE_k = 7
    user_models = {}  # this is a dictionary of models for eac
    # Build biometric models for each user
    # There would two models for each user, namely, landscape and portrait
    gen_train_x_total = []
    gen_train_y_total = []
    for user_id in range(180, 187):
        if user_id != 187:
            print(user_id)
            session, gen_train_x, imp_train_x, gen_test, imp_test = get_user_model(user_id)
            print("User" + str(user_id) + "_" + session)


            training_x = gen_train_x + imp_train_x
            training_y = []
            for i in range(0, len(gen_train_x)):
                training_y.append(1)
            for i in range(0, len(imp_train_x)):
                training_y.append(0)

            gen_train_x_total.append(training_x)
            gen_train_y_total.append(training_y)

    # print("Correcting Oversampling...")
    # oversampling = SMOTE(sampling_strategy=1.0, random_state=random_seed, k_neighbors=SMOTE_k)
    # training_x, training_y = oversampling.fit_resample(np.array(gen_train_x_total), np.array(gen_train_y_total))

    print("Selecting Features...")
    training_x, training_y, gen_test, imp_test = select_features(gen_train_x_total, gen_train_y_total,
                                                             gen_test,
                                                             imp_test,
                                                             feature_selection_threshold)

    print("Getting Error Rates...")
    tn, fp, fn, tp = get_error_rates(training_x, training_y, gen_test, imp_test, "kNN")

    far = fp / (fp + tn)
    frr = fn / (fn + tp)
    tar = tp / (fn + tp)
    trr = tn / (fp + tn)
    hter = (far + frr) / 2

    print("HTER:", hter)

    analyse_results(far, frr, tar, trr)

    user_result = ["User"+str(user_id), session, "kNN", tn, fp, fn, tp, hter]
    final_result.append(user_result)

    result_dataframe = pd.DataFrame(final_result, columns=['user', 'mode', 'method', 'tn', 'fp', 'fn', 'tp', "HTER"])
    result_dataframe.to_csv("final_result.csv", encoding='utf-8', index=False)

# for method in classification_method:
## Your basic job is to create the following matrices
# training_X: this is training feature matrix that contains feature vectors from genuine and impostors
# The genuine feature vectors are created from the data of genuine user collected in Session1
# Similarly the impostor feature vectors are created from the data of users other than the genuine user collected in Session1
# training_y: this is basically a column of ONES (genuine) and ZEROES (impostors)
# genuine_testing: feature matrix that consists of feature vectors created from the data of genuine user collected in Session2
# impostor_testing: feature matrix that consists of feature vectors created from the data of users other than the genuine user collected in Session2

# Apply SMOTE if needed to address the class imbalance problem
# following is the code that you can use. try other variants of SMOTE if you think that is needed
# oversampling = SMOTE(sampling_strategy=1.0, random_state=random_seed,k_neighbors=SMOTE_k)
# training_X, training_y = oversampling.fit_resample(training_X, training_y)

# Select the features before running the classification, for example using mutual information
# training_X, training_y, genuine_testing, impostor_testing = select_features(training_X, training_y,
#                                                                             genuine_testing,
#                                                                             impostor_testing,
#                                                                             feature_selection_threshold)

#
#

#

#
# training_x, training_y, gen_test, imp_test = select_features(training_x, training_y,
#                                                              gen_test_x,
#                                                              imp_test_x,
#                                                              feature_selection_threshold)
#
# # Find the error rate using a classification method and save the errors
# tn, fp, fn, tp = get_error_rates(training_x, training_y, gen_test, imp_test, method)
# row_counter = row_counter + 1
# final_result.loc[row_counter] = [user, mode, method, tn, fp, fn, tp]
