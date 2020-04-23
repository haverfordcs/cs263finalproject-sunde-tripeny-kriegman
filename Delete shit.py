import os


if __name__ == "__main__":
    file_1 = "_Desktop_Keyboard.csv"
    file_2 = "_HandPhone_Accelerometer_(HTC_One).csv"
    file_3 = "_HandPhone_Checkpoints_(HTC_One).csv"
    file_4 = "_HandPhone_Gyroscope_(HTC_One).csv"
    file_5 = "_HandPhone_Keyboard_(HTC_One).csv"
    file_6 = "_HandPhone_TouchEvent_(HTC_One).csv"  # keep
    file_7 = "_HandTablet_Accelerometer_(Nexus9).csv"
    file_8 = "_HandTablet_Checkpoints_(Nexus9).csv"
    file_9 = "_HandTablet_Gyroscope_(Nexus9).csv"
    file_10 = "_HandTablet_Keyboard_(Nexus9).csv"
    file_11 = "_HandTablet_TouchEvent_(Nexus9).csv"
    file_12 = "_Mouse_Button.csv"
    file_13 = "_Mouse_Move.csv"
    file_14 = "_Mouse_Wheel.csv"
    file_15 = "_PocketPhone_Accelerometer_(Samsung_S6).csv"  # keep
    file_16 = "_PocketPhone_Gyroscope_(Samsung_S6).csv"  # keep

    data_path = "BB-MAS_Dataset\\"
    for i in range(1, 118):
        to_delete = data_path + str(i) + "\\" + str(i) + file_1
        if os.path.exists(to_delete):
            os.remove(to_delete)
        to_delete = data_path + str(i) + "\\" + str(i) + file_2
        if os.path.exists(to_delete):
            os.remove(to_delete)
        to_delete = data_path + str(i) + "\\" + str(i) + file_3
        if os.path.exists(to_delete):
            os.remove(to_delete)
        to_delete = data_path + str(i) + "\\" + str(i) + file_4
        if os.path.exists(to_delete):
            os.remove(to_delete)
        to_delete = data_path + str(i) + "\\" + str(i) + file_5
        if os.path.exists(to_delete):
            os.remove(to_delete)
        to_delete = data_path + str(i) + "\\" + str(i) + file_7
        if os.path.exists(to_delete):
            os.remove(to_delete)
        to_delete = data_path + str(i) + "\\" + str(i) + file_8
        if os.path.exists(to_delete):
            os.remove(to_delete)
        to_delete = data_path + str(i) + "\\" + str(i) + file_9
        if os.path.exists(to_delete):
            os.remove(to_delete)
        to_delete = data_path + str(i) + "\\" + str(i) + file_10
        if os.path.exists(to_delete):
            os.remove(to_delete)
        to_delete = data_path + str(i) + "\\" + str(i) + file_11
        if os.path.exists(to_delete):
            os.remove(to_delete)
        to_delete = data_path + str(i) + "\\" + str(i) + file_12
        if os.path.exists(to_delete):
            os.remove(to_delete)
        to_delete = data_path + str(i) + "\\" + str(i) + file_13
        if os.path.exists(to_delete):
            os.remove(to_delete)
        to_delete = data_path + str(i) + "\\" + str(i) + file_14
        if os.path.exists(to_delete):
            os.remove(to_delete)

    pass
