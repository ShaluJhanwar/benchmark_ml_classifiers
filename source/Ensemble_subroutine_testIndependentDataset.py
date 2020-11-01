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
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
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
    y_data = np.genfromtxt(data_file,  usecols = label_col, delimiter = "\t", skiprows=1)
    test_x_data = np.loadtxt(test_file, usecols=data_cols, delimiter = "\t", skiprows=1)
    test_y_data = np.genfromtxt(test_file,  usecols = label_col, delimiter = "\t", skiprows=1)
    
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
        b = np.genfromtxt(StringIO(a), usecols = cols, delimiter="\t", dtype=None)
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
    plt.plot(fpr, tpr, 'k--',label='RF_ROC_all_data (area = %0.2f)' % random_mean_auc, lw=3, color=(0.45, 0.42, 0.18)) #Plot mean ROC area in cross validation
    #plt.plot(fpr_FS, tpr_FS, 'k--',label='Random Forest (area = %0.2f)' % random_mean_auc, lw=2, color=(0.93, 0.12, 0.78)) #Plot mean ROC area in cross validation
    #plt.plot(fpr_FS, tpr_FS, 'k--',label='SVM (area = %0.2f)' % random_mean_auc, lw=2, color=(0.43, 0.82, 0.68)) #Plot mean ROC area in cross validation
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    return plt


class EnsembleClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    """ Soft Voting/Majority Rule classifier for unfitted clfs.

    Parameters
    ----------
    clfs : array-like, shape = [n_classifiers]
      A list of classifiers.
      Invoking the `fit` method on the `VotingClassifier` will fit clones
      of those original classifiers that will be stored in the class attribute
      `self.clfs_`.

    voting : str, {'hard', 'soft'} (default='hard')
      If 'hard', uses predicted class labels for majority rule voting.
      Else if 'soft', predicts the class label based on the argmax of
      the sums of the predicted probalities, which is recommended for
      an ensemble of well-calibrated classifiers.

    weights : array-like, shape = [n_classifiers], optional (default=`None`)
      Sequence of weights (`float` or `int`) to weight the occurances of
      predicted class labels (`hard` voting) or class probabilities
      before averaging (`soft` voting). Uses uniform weights if `None`.

    Attributes
    ----------
    classes_ : array-like, shape = [n_predictions]

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> clf1 = LogisticRegression(random_state=1)
    >>> clf2 = RandomForestClassifier(random_state=1)
    >>> clf3 = GaussianNB()
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> eclf1 = VotingClassifier(clfs=[clf1, clf2, clf3], voting='hard')
    >>> eclf1 = eclf1.fit(X, y)
    >>> print(eclf1.predict(X))
    [1 1 1 2 2 2]
    >>> eclf2 = VotingClassifier(clfs=[clf1, clf2, clf3], voting='soft')
    >>> eclf2 = eclf2.fit(X, y)
    >>> print(eclf2.predict(X))
    [1 1 1 2 2 2]
    >>> eclf3 = VotingClassifier(clfs=[clf1, clf2, clf3],
    ...                          voting='soft', weights=[2,1,1])
    >>> eclf3 = eclf3.fit(X, y)
    >>> print(eclf3.predict(X))
    [1 1 1 2 2 2]
    >>>
    """
    def __init__(self, clfs, voting='hard', weights=None):

        self.clfs = clfs
        self.named_clfs = {key:value for key,value in _name_estimators(clfs)}
        self.voting = voting
        self.weights = weights


    def fit(self, X, y):
        """ Fit the clfs.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object
        """
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output'\
                                      ' classification is not supported.')

        if self.voting not in ('soft', 'hard'):
            raise ValueError("Voting must be 'soft' or 'hard'; got (voting=%r)"
                             % voting)

        if self.weights and len(self.weights) != len(self.clfs):
            raise ValueError('Number of classifiers and weights must be equal'
                             '; got %d weights, %d clfs'
                             % (len(self.weights), len(self.clfs)))

        self.le_ = LabelEncoder()
        self.le_.fit(y)
        self.classes_ = self.le_.classes_
        self.clfs_ = []
        for clf in self.clfs:
            fitted_clf = clone(clf).fit(X, self.le_.transform(y))
            self.clfs_.append(fitted_clf)
        return self

    def predict(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        maj : array-like, shape = [n_samples]
            Predicted class labels.
        """
        if self.voting == 'soft':

            maj = np.argmax(self.predict_proba(X), axis=1)

        else:  # 'hard' voting
            predictions = self._predict(X)

            maj = np.apply_along_axis(
                                      lambda x:
                                      np.argmax(np.bincount(x,
                                                weights=self.weights)),
                                      axis=1,
                                      arr=predictions)

        maj = self.le_.inverse_transform(maj)
        return maj

    def predict_proba(self, X):
        """ Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        avg : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.
        """
        avg = np.average(self._predict_probas(X), axis=0, weights=self.weights)
        return avg

    def transform(self, X):
        """ Return class labels or probabilities for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        If `voting='soft'`:
          array-like = [n_classifiers, n_samples, n_classes]
            Class probabilties calculated by each classifier.
        If `voting='hard'`:
          array-like = [n_classifiers, n_samples]
            Class labels predicted by each classifier.
        """
        if self.voting == 'soft':
            return self._predict_probas(X)
        else:
            return self._predict(X)

    def get_params(self, deep=True):
        """ Return estimator parameter names for GridSearch support"""
        if not deep:
            return super(EnsembleClassifier, self).get_params(deep=False)
        else:
            out = self.named_clfs.copy()
            for name, step in six.iteritems(self.named_clfs):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out

    def _predict(self, X):
        """ Collect results from clf.predict calls. """
        return np.asarray([clf.predict(X) for clf in self.clfs_]).T

    def _predict_probas(self, X):
        """ Collect results from clf.predict calls. """
        return np.asarray([clf.predict_proba(X) for clf in self.clfs_])

if __name__ == "__main__":
    main(sys.argv[1:])
