# Real-time-Arryhthmia-Detector

This repository contains the implementation of the Research Paper 
### "Real Time Arrhythmia Detecting Wearable using a Novel Deep Learning Model”
authored by Sayak Banerjee, Arin Paul, Anshika Agarwal and Sumit Kumar Jindal
and presented at 
### International Conference on Interdisciplinary Cyber Physical Systems, ICPS 2020 held at IIT Madras



For the implementation of the deep learning model the Physionet Challenge 2017 CinC dataset has been used. The dataset houses 8528 samples of single-lead short ECG signals. The ECG signals have been sampled with a sampling frequency of 300Hz. The duration of each of the signal varies between 20s and 1 min. The dataset has been divided into four different classes of ECG samples. There are 5154 samples belonging to the normal sinus rhythm class while 771 samples have been allotted to atrial fibrillation class. Furthermore, 2557 samples of ECG are labelled as others and only 46 samples out of the available 8528 samples are noisy.


## Result

The training accuracy is 94.95%. F1 score parameters have been calculated on the test set of the 2017 Physionet Computing in Cardiology challenge dataset. F1n score of ~91%, F1a score of ~79%, F1o and F1p scores of ~73% with an overall F1 score of 81% is obtained from the model. The accuracy score for the test set is 84.64%.


The code is now publicly available.
