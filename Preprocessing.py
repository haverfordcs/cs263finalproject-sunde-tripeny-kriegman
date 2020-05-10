# Install packages as per the need
import os
import pandas as pd
import pickle as pc


def swipe_preprocessor(data_path, path_type):
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


def gait_preprocessor(users, segments):
    for user in users:
        print("Preprocessing User" + str(user))
        a_path = "BB-MAS_Dataset/" + str(user) + "/" + str(user) + "_PocketPhone_Accelerometer_(Samsung_S6).csv"
        g_path = "BB-MAS_Dataset/" + str(user) + "/" + str(user) + "_PocketPhone_Gyroscope_(Samsung_S6).csv"

        if os.path.exists(a_path) and os.path.exists(g_path):
            a_data_frame = pd.read_csv(a_path)
            g_data_frame = pd.read_csv(g_path)

            a_data_frame = smooth(a_data_frame, 1)
            g_data_frame = smooth(g_data_frame, 1)

            a_segments = split(a_data_frame, segments)
            g_segments = split(g_data_frame, segments)

            length = len(a_segments[0])
            for i in range(0, len(a_segments)):
                a_segments[i].index = range(length)

            length = len(g_segments[0])
            for i in range(0, len(g_segments)):
                g_segments[i].index = range(length)

            path = "StepDataByUser\\User" + str(user)
            if not os.path.exists(path):
                os.makedirs(path)

            # serialize all the data for each user
            pickle_out = open(path + "\\" + "accelerometer", "wb")
            pc.dump(a_segments, pickle_out)
            pickle_out.close()

            pickle_out = open(path + "\\" + "gyroscope", "wb")
            pc.dump(g_segments, pickle_out)
            pickle_out.close()
        else:
            print("User" + str(user) + " invalid!")


def split(data, seg_length):
    length = len(data)
    chunks = int(length / seg_length)
    segments = []

    for i in range(len(data)):
        sublist = data[seg_length * i:seg_length + seg_length * i]
        if len(sublist) == seg_length:
            segments.append(sublist)
        else:
            break

    print(len(segments))
    return segments


def smooth(data, iterations):
    smooth_data = data.copy()
    for key in ["Xvalue", "Yvalue", "Zvalue"]:
        # run through each sequence of 3 points and average the middle point with the outside two
        for j in range(0, iterations):
            for i in range(0, (len(smooth_data) - 2)):
                smooth_data[key][i + 1] = (smooth_data[key][i] + smooth_data[key][i + 1] + smooth_data[key][i + 2]) / 3
    return smooth_data

    #
    #
    # group = data_frame["UserID"]
    # agg = data_frame.groupby([group])
    # for user in agg:
    #     # make new path if one doesn't already exist
    #     user_swipes = []
    #     group = user[1]["SwipeID"]
    #     agg1 = user[1].groupby([group])
    #     for swipe in agg1:
    #         swipe[1].index = range(len(swipe[1].index))
    #         user_swipes.append(swipe[1])
    #
    #     user_id = user[0]
    #     path = "DataByUser\\User" + str(user_id)
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #
    #     # serialize all the data for each user
    #     pickle_out = open(path + "\\" + path_type, "wb")
    #     pc.dump(user_swipes, pickle_out)
    #     pickle_out.close()
    #
    #     print(user_swipes)
    # pass


if __name__ == "__main__":
    gait_preprocessor(range(1, 118), 1000)
    pass
