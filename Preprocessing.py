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

            a_segments = split(a_data_frame, segments)
            g_segments = split(a_data_frame, segments)

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


def split(data, chunks):
    length = len(data)
    seg_length = int(length / chunks)
    segments = [data[x:x + seg_length] for x in range(0, length - length % chunks, seg_length)]
    return segments

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
    preprocess(range(1, 119), 4)
    pass
