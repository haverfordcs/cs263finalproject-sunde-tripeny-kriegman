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



def preprocess(data_path, path_type):
    data_frame = pd.read_csv(data_path)

    for user in range(1, 117):
        a_path = str(user) + "\\" + str(user) + "_PocketPhone_Accelerometer_(Samsung_S6).csv"
        g_path = str(user) + "\\" + str(user) + "_PocketPhone_Gyroscope_(Samsung_S6).csv"
        if os.path.exists(a_path) and os.path.exists(g_path):
            a_data_frame = pd.read_csv(a_path)
            g_data_frame = pd.read_csv(g_path)




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


if __name__ == "__main__":
    pass