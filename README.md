This project is an extension of labs 2, 3, and 5. In it, we use both gate and swiping biometrics in an to
create a multi-modal continuous authentication system.

----Contents----
-BB-MAS_Dataset: Contains the raw data for gait authentication by user. The data is only for walking on flat ground.
-FeatureByUser: Contains pickled, extracted features for swipe authentication.
-Results: Contains .csv files with the results of both GaitFeatureExtraction.py and SwipeFeatureExtraction.py.
-StepDataByUser: Contains pickled, preprocessed data for gait authentication by user.
-SwipeData: Contains raw data for swipe authentication in .csv files by sensor.
-SwipeDataByUser: Contains preprocessed swipe data organized by user.
-Final Project writeup.pdf: The writeup for this project.
-GaitFeatureExtraction.py: Contains the methods to fit the data in StepDataByUser to a model and analyze it. Produces the
 results found in the Results folder.
-lab_5_sunde_tripeny.py: Contains lab 5 to use as reference for swipe authentication.
-Preprocessing.py: Contains the preprocessing methods for both gait and swipe feature extraction.
-README.md: You're reading it, so hopefully you know what it's for.
-SwipeFeatureExtraction.py: Contains the methods to produce the pickled features in FeaturByUser and analyze them to
 produce the result in Results folder.

----Documentation----
pandas: https://pandas.pydata.org/docs/
pickle: https://docs.python.org/3/library/pickle.html
sklearn: https://scikit-learn.org/stable/index.html
fastdtw: https://pypi.org/project/fastdtw/

 ----Sources----
[1] https://arxiv.org/pdf/1501.01199.pdf Benchmarking Touchscreen Biometrics for Mobile Authentication
[2] I. C. Stylios, O. Thanou, I. Androulidakis, and  E. Zaitseva, “A review of continuous authentication using
    behavioral biometrics,” in ACM International Conference Proceeding Series, 2016, vol. 25-27-September-2016, pp. 72–79.
[3] Amith K. Belman, Li Wang, Sundaraja S. Iyengar, Pawel Sniatala, Robert Wright, Robert Dora, Jacob Baldwin, Zhanpeng
    Jin, Vir V. Phoha, "SU-AIS BB-MAS (Syracuse University and Assured Information Security - Behavioral Biometrics
    Multi-device and multi-Activity data from Same users) Dataset ", IEEE Dataport, 2019. [Online]. Available:
    http://dx.doi.org/10.21227/rpaz-0h66. Accessed: Apr. 28, 2020.
