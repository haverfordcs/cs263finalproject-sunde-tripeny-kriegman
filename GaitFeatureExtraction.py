import os
import pandas
import pickle as pc
import numpy as np
from fastdtw import fastdtw as fdtw
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


class Params:
    file_types = ["accelerometer", "gyroscope"]
    valid_keys = ["Xvalue", "Yvalue", "Zvalue"]
    num_users = 117
    evaluation_file = "Results/GaitAuthenticationResults.txt"


# custom metric
def DTW(a, b):
    distance, path = fdtw(a, b)

    return distance


if __name__ == "__main__":
    X = []
    y = []

    print("Getting User Data")
    for user in range(1, 11):
        print("User" + str(user))
        acc_file = "StepDataByUser/User" + str(user) +"/accelerometer"
        gyr_file = "StepDataByUser/User" + str(user) +"/gyroscope"
        if os.path.exists(gyr_file) and os.path.exists(acc_file):
            pickle_in = open(acc_file, "rb")
            current_acc_data = pc.load(pickle_in)
            pickle_in = open(gyr_file, "rb")
            current_gyr_data = pc.load(pickle_in)
            pickle_in.close()

            for i in range(0, 10):
                temp = []
                for key in Params.valid_keys:
                    temp += current_acc_data[i][key].tolist()
                    temp += current_gyr_data[i][key].tolist()
                X.append(np.array(temp))
                y.append("User" + str(user))
        else:
            if not os.path.exists(gyr_file):
                print("Path: \'" + gyr_file + "\' Not Found!")
            else:
                print("Path: \'" + acc_file + "\' Not Found!")

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print("Split done")
    print("Fitting...")
    # train
    parameters = {'n_neighbors': [2, 4]}
    clf = GridSearchCV(KNeighborsClassifier(metric=DTW), parameters, cv=3, n_jobs=-1)
    clf.fit(X_train, y_train)

    print(clf.cv_results_)
    print("Fit Done")

    print("Starting Evaluation...")
    # evaluate
    y_pred = clf.predict(X_test)
    classification_report = classification_report(y_test, y_pred, output_dict=True)

    print(classification_report)
    temp_data_frame = pandas.DataFrame.from_dict(classification_report)
    # file = open(Params.evaluation_file, "")
    pandas.DataFrame.to_csv(temp_data_frame, Params.evaluation_file)

    print("Evaluation Saved To: " + Params.evaluation_file)
    pass
