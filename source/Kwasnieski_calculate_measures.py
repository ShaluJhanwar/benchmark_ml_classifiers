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
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix

def main(argv):
	
	try:
       		opts, args = getopt.getopt(argv, "d:c:")
        
    	except getopt.GetoptError:
	
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-d':
			data_file = arg
		elif opt == '-c':
			label_col = int(arg)
	
	y_true = np.genfromtxt(data_file,  usecols = label_col, delimiter = "\t", skip_header = 1)
	for lab in range(2,9):
		print "lab", lab
		y_pred = np.genfromtxt(data_file,  usecols = lab, delimiter = "\t", skip_header = 1)
		print "The classification report for Algorithm", lab, "is \n"
		#Make classification report
		print metrics.classification_report(y_true, y_pred)
		print "Accuracy: %.6f" %metrics.accuracy_score(y_true,y_pred)
		#Compute specificity from confusion amtrix
		cm = confusion_matrix(y_true, y_pred)
		print "Confusion matrix as \n", cm
		tn = int(cm[0,0])
		fp = int(cm[0,1])
		print "tn", tn
		print "fp", fp
		s = tn/(tn + fp)
		print "Speicificity is", s , "\n"
		print "Metthiew correlation co-efficient: %.6f" %matthews_corrcoef(y_true, y_pred)


if __name__ == "__main__":
    main(sys.argv[1:])

