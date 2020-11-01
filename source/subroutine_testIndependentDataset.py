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
import os
os.getcwd()


def main(argv):
    
    
    # get options passed at command line
    
    try:
        opts, args = getopt.getopt(argv, "d:o:c:C:t:m:")

    except getopt.GetoptError:
        
        #print helpString
        
        sys.exit(2)
#print opts
    for opt, arg in opts:
    
        if opt == '-d':
        
            data_file = arg
        
        elif opt == '-o':
            
            out_folder = arg
    
        elif opt == '-c':
            
            label_col = int(arg)
        
        elif opt == '-C':
            
            data_cols = arg
        
        elif opt == '-t':
            
            test_file = arg  #Whole genome prediction file

        elif opt == '-m':
            model_file = arg

    model_filename = os.path.abspath(model_file)
    data_file = os.path.abspath(data_file)
    test_file = os.path.abspath(test_file)
    print model_file, "\n"
    data_cols = [int(x) for x in data_cols.split(",")]
    x_data = np.loadtxt(data_file, usecols=data_cols, delimiter = "\t", skiprows=1)
    y_data = np.genfromtxt(data_file,  usecols = label_col, delimiter = "\t", skip_header=1)
    test_x_data = np.loadtxt(test_file, usecols=data_cols, delimiter = "\t", skiprows=1)
    test_y_data = np.genfromtxt(test_file,  usecols = label_col, delimiter = "\t", skip_header=1)
    
    #Load the model file#
    estimator = joblib.load(model_filename)

    #perform same scaling on training and testing data
    x_data, test_x_data = scaling_training_testing_data(x_data, test_x_data)
    np.random.seed(0)
    indices = np.random.permutation(len(test_x_data))
    test_x_data = test_x_data[indices]
    test_y_data = test_y_data[indices]
    cols = 0
    with open (test_file,"r") as temp:
        a =  '\n'.join(line.strip("\n") for line in temp)
        b = np.genfromtxt(StringIO(a), usecols = cols, delimiter="\t", dtype=None, skip_header=1)
        enhancer_names_test = b[indices]
    temp.close()
    y_FAN_pred = estimator.predict(test_x_data)
    y_score_test = estimator.predict_proba(test_x_data)
    print metrics.classification_report(test_y_data,y_FAN_pred)
    combined_test = zip(enhancer_names_test, test_y_data, y_FAN_pred, y_score_test[:,0], y_score_test[:,1])
    #f = open(out_folder + "/subroutine_RF_FANTOM_FeatureSelected_pred.txt", 'w')
    f = open(out_folder + "/GM12878_FANTOM_RF_FeatureSelected_ROC.txt", 'w')
    f.write("Enhancer_name\tY_true_labels\tY_predicted_labels\tProb_Class0\tProb_class1\n")
    for i in combined_test:
        line = '\t'.join(str(x) for x in i)
        f.write(line + '\n')
    f.close()
    print "Random Forests: On FANTOM, Final Generalization Accuracy: %.6f" %metrics.accuracy_score(test_y_data,y_FAN_pred)
    print metrics.classification_report(test_y_data,y_FAN_pred)
    print "Number of mislabeled points : %d" % (test_y_data != y_FAN_pred).sum()
    print metrics.classification_report(test_y_data,y_FAN_pred)
    print "Random Forests: Final Generalization Accuracy: %.6f" %metrics.accuracy_score(test_y_data,y_FAN_pred)
    #Before we move on, let's look at a key parameter that RF returns, namely feature_importances. This tells us which #features in our dataset seemed to matter the most (although won't matter in the present scenario with only 2 features)
    print estimator.feature_importances_

#Plot ROC#
    roc_plt = plot_roc(estimator, test_x_data, test_y_data, y_FAN_pred)
    #pl.savefig(out_folder + "/subroutine_RF_FeatureSelected_split_test_train_Kfold.svg", transparent=True, bbox_inches='tight', pad_inches=0.2)
    pl.savefig(out_folder + "/GM12878_FANTOM_RF_FeatureSelected_ROC.svg", transparent=True, bbox_inches='tight', pad_inches=0.2)
    roc_plt.show()

def scaling_training_testing_data(train_data, test_data):
    #Scaling of the data
    #min_max_scaler = preprocessing.MinMaxScaler()
    #x_data = min_max_scaler.fit_transform(x_data)
    
    #same Scaling on both test and train data (centering the data scaling)
    scaler = preprocessing.StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    return train_data, test_data


def plot_roc(estimator, test_x_data, test_y_data, y_FAN_pred):
    y_score = estimator.predict_proba(test_x_data)
    y_score =np.around(y_score, decimals=2)
    accurate = accuracy_score(test_y_data, y_FAN_pred)
    print "Accuracy All dataset: ", accurate
    prec = precision_score(test_y_data, y_FAN_pred, average='micro')
    rec = recall_score(test_y_data, y_FAN_pred, average='micro')
    fscore = fbeta_score(test_y_data, y_FAN_pred, average='micro', beta=0.5)
    areaRoc = roc_auc_score(test_y_data, y_score[:,1])
    
    #Generate ROC curve for each cross-validation
    fpr, tpr, thresholds = roc_curve(test_y_data, y_score[:,1], pos_label = 1)  #Pos level for positive class
    precision, recall, threshold = precision_recall_curve(test_y_data, y_score[:,1])
    random_mean_auc = auc(fpr, tpr)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Standard')
    plt.plot(fpr, tpr, 'k--',label='RF (area = %0.2f)' % random_mean_auc, lw=3, color=(0.45, 0.42, 0.18)) #Plot mean ROC area in cross validation
    #plt.plot(fpr_FS, tpr_FS, 'k--',label='Random Forest (area = %0.2f)' % random_mean_auc, lw=2, color=(0.93, 0.12, 0.78)) #Plot mean ROC area in cross validation
    #plt.plot(fpr_FS, tpr_FS, 'k--',label='SVM (area = %0.2f)' % random_mean_auc, lw=2, color=(0.43, 0.82, 0.68)) #Plot mean ROC area in cross validation
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    return plt


if __name__ == "__main__":
    main(sys.argv[1:])
