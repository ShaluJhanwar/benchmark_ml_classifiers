__author__ = 'sjhanwar'

#!/usr/bin/python

import sys
import getopt
import numpy as np
import pylab as pl
import random
from sklearn import datasets
from sklearn.learning_curve import learning_curve
from scipy import interp
import matplotlib.pyplot as plt
from scipy import interp
from sklearn import svm
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.svm import SVC
from StringIO import StringIO
from sklearn.metrics import roc_auc_score
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import fbeta_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from sklearn.externals import joblib
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import os
os.getcwd()

def main(argv):

        data_file = "/users/so/sjhanwar/ML_tool/Resources/Improvements_2nd_round/Three_models_Genome_Epigenome_TSS/All_epigenomic_1296_HepG2_832_K562_continuous_matrix.txt" #Epigenome (no tss)
        out_folder ="/users/so/sjhanwar/ML_tool/Resources/Improvements_2nd_round/Three_models_Genome_Epigenome_TSS"
        label_col = 1
        data_cols = "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17"
        data_cols = [int(x) for x in data_cols.split(",")]
        x_data = np.loadtxt(data_file, usecols=data_cols, delimiter = "\t", skiprows = 1)
        y_data = np.genfromtxt(data_file,  usecols = label_col, delimiter = "\t", skiprows = 1)
        x_data = scaling_training_testing_data(x_data)
        np.random.seed(0)
        indices = np.random.permutation(len(x_data))
        x_data = x_data[indices]
        y_data = y_data[indices]

        X_train, y_train, X_test, y_test = make_training_testing_data(x_data, y_data, indices)
        cv = StratifiedShuffleSplit(y_train, 10, test_size=0.2, random_state=0)
        best_max_depth, best_n_estimators = grid_searchCV(cv, X_train, y_train)
        print "Best parameters are for the first: max_depth =", best_max_depth, "n_estimator:", best_n_estimators, '\n'
        estimator = RandomForestClassifier(n_estimators=best_n_estimators,max_depth=best_max_depth,random_state=0)
        #estimator = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)
        estimator.fit(X_train, y_train)
        filename = out_folder + "/Continuous_All_epigenome_feature_test.pkl"
        joblib.dump(estimator, filename, compress=9)
        #2nd genomic feature
        data_file2 = "/users/so/sjhanwar/ML_tool/Resources/Improvements_2nd_round/Three_models_Genome_Epigenome_TSS/Only_genomic_1296_HepG2_832_K562_continuous_matrix.txt"
        x_data_sec = np.loadtxt(data_file2, usecols=data_cols, delimiter = "\t", skiprows = 1)
        y_data_sec = np.genfromtxt(data_file2, usecols = label_col, delimiter = "\t", skiprows = 1)
        x_data_sec = scaling_training_testing_data(x_data_sec)
        x_data_sec = x_data_sec[indices]
        y_data_sec = y_data_sec[indices]
        X_train_sec, y_train_sec, X_test_sec, y_test_sec = make_training_testing_data(x_data_sec, y_data_sec, indices)
        cv2 = StratifiedShuffleSplit(y_train_sec, 10, test_size=0.2, random_state=0)
        best_max_depth2, best_n_estimators2 = grid_searchCV(cv2, X_train_sec, y_train_sec)
#print "Best parameters are for the second: max_depth =",best_max_depth2, "n_estimator:", best_n_estimators2, '\n'
        estimator2 = RandomForestClassifier(n_estimators=best_n_estimators2,max_depth=best_max_depth2,random_state=0)
        #estimator2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)
        estimator2.fit(X_train_sec, y_train_sec)
        filename2 = out_folder + "/continuous_Genome_without_TSS_test.pkl"
        joblib.dump(estimator2, filename2, compress=9)
        #3rd TSS only genomic feature
	data_file3 = "/users/so/sjhanwar/ML_tool/Resources/Improvements_2nd_round/Three_models_Genome_Epigenome_TSS/Only_TSS_1296_HepG2_832_K562_continuous_matrix_genomics_epigenomics.txt"
	x_data_third = np.loadtxt(data_file3, usecols=data_cols, delimiter = "\t", skiprows = 1)
        y_data_third = np.genfromtxt(data_file3, usecols = label_col, delimiter = "\t", skiprows = 1)
        x_data_third = scaling_training_testing_data(x_data_third)
        x_data_third = x_data_third[indices]
        y_data_third = y_data_third[indices]
        X_train_third, y_train_third, X_test_third, y_test_third = make_training_testing_data(x_data_third, y_data_third, indices)
        cv3 = StratifiedShuffleSplit(y_train_third, 10, test_size=0.2, random_state=0)
        best_max_depth3, best_n_estimators3 = grid_searchCV(cv3, X_train_third, y_train_third)
#print "Best parameters are for the second: max_depth =",best_max_depth2, "n_estimator:", best_n_estimators2, '\n'
        estimator3 = RandomForestClassifier(n_estimators=best_n_estimators3,max_depth=best_max_depth3,random_state=0)
        #estimator2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)
        estimator3.fit(X_train_third, y_train_third)
        filename3 = out_folder + "/continuous_Only_TSS_test.pkl"
        joblib.dump(estimator3, filename3, compress=9)	

	print "For first model\n"
        #print estimator.estimators_, "\n"

        print "Feature imp\n"
        print estimator.feature_importances_, "\n"

        print "For sec model\n"
        #print estimator2.estimators_, "\n"

        print "Feature imp\n"
        print estimator2.feature_importances_, "\n"

        #Combine function
	print "For third model\n"
        #print estimator2.estimators_, "\n"

        print "Feature imp\n"
        print estimator3.feature_importances_, "\n"

        rf_combined = combine_rfs(estimator, estimator2, estimator3)
        print "For combine model\n"
        #print rf_combined.estimators_, "\n"
        print rf_combined.feature_importances_, "\n"
        filename4 = out_folder + "/combined_continuous_Epigenome_Genome_TSS_model.pkl"
        joblib.dump(rf_combined, filename4, compress=9)

def combine_rfs(rf_a, rf_b, rf_c):
    rf_a.estimators_ += rf_b.estimators_
    rf_a.estimators_ += rf_c.estimators_
    rf_a.n_estimators = len(rf_a.estimators_)
    return rf_a


def scaling_training_testing_data(train_data):
    #Scaling of the data
    #min_max_scaler = preprocessing.MinMaxScaler()
    #x_data = min_max_scaler.fit_transform(x_data)

    #same Scaling on both test and train data (centering the data scaling)
    scaler = preprocessing.StandardScaler()
    train_data = scaler.fit_transform(train_data)
    #test_data = scaler.transform(test_data)
    return train_data

def whole_dataset_train_test(X, y):
    rfpred = RandomForestClassifier().fit(X,y)
    pred = rfpred.predict(X)
    print "When fitted on the whole dataset with selected features, then the classification report is found to be:\n";
    print "Random Forests: Accuracy: %.6f" %metrics.accuracy_score(y,pred)
    print metrics.classification_report(y, pred)

def make_training_testing_data(Xtrain, Ytrain, indices):
    indices_1 = indices[Ytrain[indices]!=0]
    indices_0 = indices[Ytrain[indices]!=1]
    #n_test = round((indices.size*.20)/2) #No. of instances for each class #SHUFFLE DATA X and Y for splitting purpose
    #n_test = int(n_test)
    #For more negative samples#
    n_test_1 = round(indices_1.size*.10) #No. of indices for class 1
    n_test_0 = round(indices_0.size*.10) #No. of indices for class 0
    test_indices=np.concatenate([(indices_1[:n_test_1]), (indices_0[:n_test_0])])
    print "The total no. of instances in straitified test samples are:", n_test_1, n_test_0 ,"\n"
    #test_indices=np.concatenate([(indices_1[:n_test]), (indices_0[:n_test])])
    
    np.random.shuffle(test_indices) 
    train_indices=np.concatenate([(indices_1[n_test_1:]), (indices_0[n_test_0:])])
    #train_indices=np.concatenate([(indices_1[n_test:]), (indices_0[n_test:])])
    np.random.shuffle(train_indices)
    X_test = Xtrain[test_indices]
    y_test = Ytrain[test_indices]
    X_train = Xtrain[train_indices]
    y_train = Ytrain[train_indices]
    print X_test.shape, y_test.shape, X_train.shape , y_train.shape
    return X_train, y_train, X_test, y_test

def grid_searchCV(cv, X_train, y_train):
    max_depth=np.linspace(1,20,30) #By default
    n_estimators=np.arange(10, 400, 10) #n_estimator
    n_jobs=16 #no_of_CPU
    clf = RandomForestClassifier(random_state=0)
    #clf = AdaBoostClassifier(random_state=0)
    classifier = GridSearchCV(estimator=clf, cv=cv, param_grid=dict(n_estimators=n_estimators, max_depth=max_depth), n_jobs=n_jobs, scoring='f1')
    #Also note that we're feeding multiple neighbors to the GridSearch to try out. #We'll now fit the training dataset to this classifier

    classifier.fit(X_train, y_train)
    #Let's look at the best estimator that was found by GridSearchCV
    print "Best Estimator learned through GridSearch for random forest are:"
    print classifier.best_estimator_ ,"\n";
    max_depth = classifier.best_estimator_.max_depth #bEST MAX_DEPTH
    n_estimators = classifier.best_estimator_.n_estimators #Best n_estimator
    return classifier.best_estimator_.max_depth, classifier.best_estimator_.n_estimators

if __name__ == "__main__":
    main(sys.argv[1:])
