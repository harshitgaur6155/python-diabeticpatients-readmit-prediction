"""
<Capstone Project - 4>

Copyright (c) 2021 -- This is the 2021 Fall A version of the Template
Licensed
Written by <Supreeth Murugesh, Harshit Gaur, Jeseeka Shah> 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn import svm

def read_dataset(url):
    """
    Read CSV File from an URL, Import data from it and assign in to an appropriate Data Structure. 
    :param url : URL
    :return: Data Structure (Data Frame)
    """
    datasetUrl = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    return pd.read_csv(datasetUrl)

def drop_irrelevant_columns(dataSet):
    """
    Drop irrelevant columns from the dataset
    :param dataSet : Data Frame
    :return: Data Structure (Data Frame)
    """

    # Removing columns as they either are IDs or unbalanced data
    columnList = ['weight', 'payer_code',  'medical_specialty', 'encounter_id', 'patient_nbr', 'discharge_disposition_id', 
                'admission_source_id', 'acetohexamide', 'tolbutamide', 'troglitazone', 'tolazamide', 'examide', 'citoglipton',
                'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone',
                'number_outpatient', 'diag_2','diag_3','race']

    for column in columnList:
        dataSet = dataSet.drop(column, axis = 1)
    
    return dataSet

def type_cast_diagnosis_to_icd9(dataSet):
    """
    Type Cast 'diagnosis' feature to their ICD-9 codes.
    :param dataSet : Data Frame
    :return: Data Structure (Data Frame)
    """

    # Converting String to Numeric - diag_1
    dataSet['diag_1'] = pd.to_numeric(dataSet['diag_1'], errors='coerce')

    # Categorizing ICD-9 codes with Float Values ranging from 1.0 to 9.0
    dataSet.loc[((dataSet.diag_1 >= 790.0) & (dataSet.diag_1 <= 799.0)) | (dataSet.diag_1 == 780.0) | (dataSet.diag_1 == 781.0) | (dataSet.diag_1 == 784.0), "diag_1"] = 9.0
    dataSet.loc[((dataSet.diag_1 >= 240.0) & (dataSet.diag_1 <= 279.0) & 
                (dataSet.diag_1 != 250.0) & (dataSet.diag_1 != 250.01) & (dataSet.diag_1 != 250.02) & (dataSet.diag_1 != 250.03) &
                (dataSet.diag_1 != 250.1) & (dataSet.diag_1 != 250.11) & (dataSet.diag_1 != 250.12) & (dataSet.diag_1 != 250.13) &
                (dataSet.diag_1 != 250.2) & (dataSet.diag_1 != 250.21) & (dataSet.diag_1 != 250.22) & (dataSet.diag_1 != 250.23) &
                (dataSet.diag_1 != 250.3) & (dataSet.diag_1 != 250.31) & (dataSet.diag_1 != 250.32) & (dataSet.diag_1 != 250.33) &
                (dataSet.diag_1 != 250.4) & (dataSet.diag_1 != 250.41) & (dataSet.diag_1 != 250.42) & (dataSet.diag_1 != 250.43) &
                (dataSet.diag_1 != 250.5) & (dataSet.diag_1 != 250.51) & (dataSet.diag_1 != 250.52) & (dataSet.diag_1 != 250.53) &
                (dataSet.diag_1 != 250.6) &
                (dataSet.diag_1 != 250.7) &
                (dataSet.diag_1 != 250.8) & (dataSet.diag_1 != 250.81) & (dataSet.diag_1 != 250.82) & (dataSet.diag_1 != 250.83) &
                (dataSet.diag_1 != 250.9) & (dataSet.diag_1 != 250.91) & (dataSet.diag_1 != 250.92) & (dataSet.diag_1 != 250.93)
                ) , "diag_1"] = 9.0
    dataSet.loc[((dataSet.diag_1 >= 680.0) & (dataSet.diag_1 <= 709.0)) | (dataSet.diag_1 == 782.0), "diag_1"] = 9.0
    dataSet.loc[(dataSet.diag_1 >= 001.0) & (dataSet.diag_1 <= 139.0), "diag_1"] = 9.0
    dataSet.loc[(dataSet.diag_1 >= 290.0) & (dataSet.diag_1 <= 319.0), "diag_1"] = 9.0
    dataSet.loc[(dataSet.diag_1 >= 280.0) & (dataSet.diag_1 <= 289.0), "diag_1"] = 9.0
    dataSet.loc[(dataSet.diag_1 >= 320.0) & (dataSet.diag_1 <= 359.0), "diag_1"] = 9.0
    dataSet.loc[(dataSet.diag_1 >= 630.0) & (dataSet.diag_1 <= 679.0), "diag_1"] = 9.0
    dataSet.loc[(dataSet.diag_1 >= 360.0) & (dataSet.diag_1 <= 389.0), "diag_1"] = 9.0
    dataSet.loc[(dataSet.diag_1 >= 740.0) & (dataSet.diag_1 <= 759.0), "diag_1"] = 9.0
    dataSet.loc[(dataSet.diag_1 == 789.0) | (dataSet.diag_1 == 783.0), "diag_1"] = 9.0

    dataSet.loc[((dataSet.diag_1 >= 390.0) & (dataSet.diag_1 <= 459.0)) | (dataSet.diag_1 == 785.0), "diag_1"] = 1.0 # Circulatory
    dataSet.loc[((dataSet.diag_1 >= 460.0) & (dataSet.diag_1 <= 519.0)) | (dataSet.diag_1 == 786.0), "diag_1"] = 2.0 # Respiratory
    dataSet.loc[((dataSet.diag_1 >= 520.0) & (dataSet.diag_1 <= 579.0)) | (dataSet.diag_1 == 787.0), "diag_1"] = 3.0 # Digestive

    dataSet.loc[(dataSet.diag_1 == 250.0) | (dataSet.diag_1 == 250.01) | (dataSet.diag_1 == 250.02) | (dataSet.diag_1 == 250.03) |
                (dataSet.diag_1 == 250.1) | (dataSet.diag_1 == 250.11) | (dataSet.diag_1 == 250.12) | (dataSet.diag_1 == 250.13) |
                (dataSet.diag_1 == 250.2) | (dataSet.diag_1 == 250.21) | (dataSet.diag_1 == 250.22) | (dataSet.diag_1 == 250.23) |
                (dataSet.diag_1 == 250.3) | (dataSet.diag_1 == 250.31) | (dataSet.diag_1 == 250.32) | (dataSet.diag_1 == 250.33) |
                (dataSet.diag_1 == 250.4) | (dataSet.diag_1 == 250.41) | (dataSet.diag_1 == 250.42) | (dataSet.diag_1 == 250.43) |
                (dataSet.diag_1 == 250.5) | (dataSet.diag_1 == 250.51) | (dataSet.diag_1 == 250.52) | (dataSet.diag_1 == 250.53) |
                (dataSet.diag_1 == 250.6) |
                (dataSet.diag_1 == 250.7) |
                (dataSet.diag_1 == 250.8) | (dataSet.diag_1 == 250.81) | (dataSet.diag_1 == 250.82) | (dataSet.diag_1 == 250.83) |
                (dataSet.diag_1 == 250.9) | (dataSet.diag_1 == 250.91) | (dataSet.diag_1 == 250.92) | (dataSet.diag_1 == 250.93), "diag_1"] = 4.0 # Diabetes

    dataSet.loc[(dataSet.diag_1 >= 800.0) & (dataSet.diag_1 <= 999.0), "diag_1"] = 5.0 # Injury
    dataSet.loc[(dataSet.diag_1 >= 710.0) & (dataSet.diag_1 <= 739.0), "diag_1"] = 6.0 # Musculoskeletal
    dataSet.loc[((dataSet.diag_1 >= 580.0) & (dataSet.diag_1 <= 629.0)) | (dataSet.diag_1 == 788.0), "diag_1"] = 7.0 # Genitourinary
    dataSet.loc[(dataSet.diag_1 >= 140.0) & (dataSet.diag_1 <= 239.0), "diag_1"] = 8.0 # Neoplasms

    evList = ['E909', 'V07', 'V25', 'V26', 'V43', 'V45', 'V51', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V60', 'V63', 'V66', 'V67', 'V70', 'V71']
    dataSet.loc[dataSet['diag_1'].isin(evList), "diag_1"] = 9.0
    return dataSet

def label_target_column(dataSet):
    """
    Label (Classify) the target column properly into binary values.
    :param dataSet : Data Frame
    :return: Data Structure (Data Frame)
    """

    # Classifying readmission values
    dataSet.loc[(dataSet.readmitted == ">30"), "readmitted"] = "YES" 
    dataSet.loc[(dataSet.readmitted == "<30"), "readmitted"] = "YES"
    return dataSet

def one_hot_encoding(dataSet, columnList):
    """
    Perform One-Hot Encoding in the dataset for categorical features.
    :param dataSet : Data Frame, columnList : List of columns
    :return: Data Structure (Data Frame)
    """

    # Creating Dummy features and droping the original features from data set.
    for i in columnList:
        if i in dataSet.columns:
            oneHot = pd.get_dummies(dataSet[i], prefix=i)
            dataSet = dataSet.join(oneHot)
            dataSet = dataSet.drop(i,axis = 1)
    return dataSet

def set_target_value(dataSet):
    """
    Set the Target Value for models. Drop this feature from the dataset that will be imported for modelling.
    :param dataSet : Data Frame
    :return: Data Structure (Data Frame)
    """

    # Setting 'Target Value' for Training. Removing the feature from the data set.
    targetValueIndex = dataSet.columns.get_loc('readmitted')
    targetValues = dataSet.iloc[: , targetValueIndex]
    dataSet = dataSet.drop('readmitted',axis = 1)
    return [dataSet, targetValues]

def drop_unbalanced_columns(dataSet):
    """
    Drop unbalanced features from the dataset.
    :param dataSet : Data Frame
    :return: Data Structure (Data Frame)
    """
    datasetUrl = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    return pd.read_csv(datasetUrl)

def plot_age_readmitted(dataSet):
    """
    Plot a graph for 'Age vs Readmitted' from the dataset
    :param dataSet : Data Frame
    :return: PLT figure
    """
    #Plotting the AGE vs READMITTANCE (Grouped by Age)
    x = dataSet.groupby(['age'],as_index=False).count()
    figure = plt.figure(figsize=(12,4))
    plt.bar(x['age'],height =x['readmitted']) 
    plt.xlabel('Age')  
    plt.ylabel('count')
    plt.title('Age vs Readmitted')
    return figure

def plot_gender_readmitted(dataSet):
    """
    Plot a graph for 'Gender vs Readmitted' from the dataset
    :param dataSet : Data Frame
    :return: PLT figure
    """
    #Plotting the AGE vs READMITTANCE (Grouped by Age)
    figure = plt.figure(figsize=(10,8))
    Z = dataSet.groupby(['gender','readmitted'],as_index=False).count() 
    t1 = Z[Z['gender']=='Male']
    t2 = Z[Z['gender']=='Female']

    X = t1['readmitted']
    Y = t1['encounter_id']
    Z = t2['encounter_id']
    df = pd.DataFrame(np.c_[Y,Z], index=X)
    df.plot.bar()

    plt.xlabel('Readmittance')  
    plt.ylabel('Count of patients')
    plt.legend(['Male', 'Female'])
    plt.title('Gender vs Readmitted')
    return figure

def plot_gender_a1cresult(dataSet):
    """
    Plot a graph for 'Age vs A1Cresult' from the dataset
    :param dataSet : Data Frame
    :return: PLT figure
    """
    #Plotting the AGE vs READMITTANCE (Grouped by Age)
    figure = plt.figure(figsize=(10,8))
    Z1 = dataSet.groupby(['gender','A1Cresult'],as_index=False).count()
    t1 = Z1[Z1['gender']=='Male']
    t2 = Z1[Z1['gender']=='Female']

    X = t1['A1Cresult']
    Y = t2['encounter_id']
    Z = t1['encounter_id']
    df = pd.DataFrame(np.c_[Y,Z], index=X)
    df.plot.bar()
    plt.title("A1C values at the time of admission for Male vs Female")
    plt.ylabel("Count")
    plt.legend('MF')
    return figure

def plot_medication_patients(dataSet):
    """
    Plot a graph for 'Count of Patients vs Change in Medications' from the dataset
    :param dataSet : Data Frame
    :return: PLT figure
    """
    #Plotting the AGE vs READMITTANCE (Grouped by Age)
    figure = plt.figure()
    Z = dataSet.groupby(['change'],as_index=False).count()
    t1 = Z[Z['change']=='Ch']
    plt.bar(Z.change, Z.age, color='LIGHTBLUE') 
    plt.xlabel('Change in the Medication')  
    plt.ylabel('Count of Patients')
    plt.title("Count of Change in Medication")
    return figure

def plot_hba1c_patients(dataSet):
    """
    Plot a graph for 'Patients Category vs HbA1c Result' from the dataset
    :param dataSet : Data Frame
    :return: PLT figure
    """
    #Plotting the AGE vs READMITTANCE (Grouped by Age)
    x1 = dataSet.groupby(['A1Cresult']).agg({'number_inpatient':'sum','number_outpatient':'sum','number_emergency':'sum'}).reset_index()
    figure = x1[x1['A1Cresult']!='None'].plot(x='A1Cresult',
        kind='bar',
        stacked=False,
        ylabel = 'Count of Patients',
        title='Patient Category vs HbA1c Result',
        figsize = (10,6),rot=0)
    return figure

def plot_admission_A1C_value(dataSet):
    """
    Plot a graph for 'Admission Status vs A1C Value' from the dataset
    :param dataSet : Data Frame
    :return: PLT figure
    """

    Z1 = dataSet.groupby(['readmitted','A1Cresult'], as_index=False).count() 
    t1 = Z1[Z1['readmitted']=='YES']
    t2 = Z1[Z1['readmitted']=='NO']

    X = t1['A1Cresult']
    Y = t2['admission_type_id']
    Z = t1['admission_type_id']

    df = pd.DataFrame(np.c_[Y,Z], index=X)
    figure = plt.figure()
    df.plot.bar()
    plt.title("A1C values at the time of admission for Male vs Female")
    plt.ylabel("Count")
    plt.legend(['Not Readmitted', 'Readmitted'])
    return figure

def random_forest_classifier(X_train, X_test, y_train, y_test):
    """
    Implement Random Forest Classifier Model.
    :param X_train, X_test, y_train, y_test : Training & Testing Data Frames
    :return: List of dictionary, error, classifier
    """

    ###################  Calculating n-Estimators and corresponding Mean Errors ###################
    error = []

    resultDict = {}
    estimator = range(100, 700, 50)
    for i in estimator:
        clf = RandomForestClassifier(n_estimators = i)
        #Train the model using the training sets
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        # Calculating error for N-Estimator values between 100 and 700
        error.append(np.mean(y_pred != y_test))
        resultDict[i] = round(metrics.accuracy_score(y_test, y_pred),3)
        
    return [resultDict, error, clf]

def plot_random_forest_mean_error(error):
    """
    Plot a graph for 'Mean Error Values' for the model.
    :param error : List of error values
    :return: PLT figure
    """
    
    ###################  Plotting n-Estimators and corresponding mean errors ###################
    figure = plt.figure()
    plt.figure(figsize=(12, 10))
    plt.plot(range(100, 700, 50), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
    plt.title('Error Rate vs n-Estimator')
    plt.xlabel('n-Estimator')
    plt.ylabel('Mean Error')
    return figure

def knn_classifier(X_train, X_test, y_train, y_test):
    """
    Implement K-Nearest Neighbor Classifier Model.
    :param X_train, X_test, y_train, y_test : Training & Testing Data Frames
    :return: List of dictionary, error, Confusion Matrix
    """

    ###################  Calculating K-Neighbors and corresponding Mean Errors ###################
    error = []

    resultDict = {}
    n_neighbors = [4, 6, 8, 10, 12 ,14, 16, 18]
    for i in n_neighbors:
        knn = KNeighborsClassifier(n_neighbors = i)
        #Train the model using the training sets
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        # Calculating error for K-Neighbor values between 4 and 18
        error.append(np.mean(y_pred != y_test))
        resultDict[i] = round(metrics.accuracy_score(y_test, y_pred),3)
        # Calculating confusion matrix
        cf_matrix = confusion_matrix(y_test, y_pred)

    return [resultDict, error, cf_matrix]

def plot_knn_mean_error(error):
    """
    Plot a graph for 'Mean Error Values' for the model.
    :param error : List of error values
    :return: PLT figure
    """

    ###################  Plotting K-Neighbors and corresponding Mean Errors ###################
    figure = plt.figure()
    plt.figure(figsize=(20, 20))
    plt.plot(range(4, 20, 2), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    return figure

def logistic_regression(X_train, X_test, y_train, y_test):
    """
    Implement Logistic Regression Model.
    :param X_train, X_test, y_train, y_test : Training & Testing Data Frames
    :return: List of accuracies, Confusion Matrix
    """

    logisticRegression = LogisticRegression()
    logisticRegression.fit(X_train, y_train)
    y_pred = logisticRegression.predict(X_test)
    accuracy = logisticRegression.score(X_test, y_test)
    # Calculating confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    return [accuracy, cf_matrix]

def svm_classifier(X_train, X_test, y_train, y_test):
    """
    Implement Support Vector Machine Classifier Model.
    :param X_train, X_test, y_train, y_test : Training & Testing Data Frames
    :return: List of result dictionary, accuracy Dictionary, Confusion Matrix
    """

    resultDict = {}
    accuracyDict = {}
    kernelList = ['linear', 'poly']
    for i in kernelList:
        clf = svm.SVC(kernel = i)
        #Train the model using the training sets
        clf.fit(X_train, y_train)
        #Predict the response for test dataset
        y_pred = clf.predict(X_test)
        resultDict[i] = classification_report(y_test, y_pred)
        accuracyDict[i] = metrics.accuracy_score(y_test, y_pred)
        cf_matrix = confusion_matrix(y_test, y_pred)
    return [resultDict, accuracyDict, cf_matrix]

if __name__ == '__main__':
    print("Initializing")
    
    