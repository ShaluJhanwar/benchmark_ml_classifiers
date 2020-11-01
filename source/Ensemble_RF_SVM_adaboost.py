__author__ = 'sjhanwar'

#!/usr/bin/python
import os
import sys
import getopt
import numpy as np
import pylab as pl
import random
import operator
from scipy import interp
import matplotlib.pyplot as plt
from StringIO import StringIO

from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from sklearn import datasets
from sklearn.learning_curve import learning_curve
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV

from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import metrics, roc_curve, auc, precision_recall_curve, fbeta_score, accuracy_score, roc_auc_score, recall_score, precision_score

from sklearn.externals import joblib
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators

os.getcwd()

def main(argv):

    np.random.seed(123)
    data_file = "/users/so/sjhanwar/ML_tool/Resources/Improvements_2nd_round/1296_HepG2_832_K562_continuous_matrix_genomics_epigenomics.txt" #Epigenome (with tss)
    out_folder ="/users/so/sjhanwar/ML_tool/Resources/Improvements_2nd_round/threshold_RF_Ensemble_voting_classifiers"
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

    svm_classifier = SVC(random_state=0)
    random_forest_classifier = RandomForestClassifier(random_state = 0)
    adaboost_classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), random_state=0)
    n_jobs=16 #no_of_CPU
    #For SVM
    C_range = np.logspace(-2, 10, 3)
    gamma_range = np.logspace(-9, 3, 3)
    param_grid = dict(gamma=gamma_range, C=C_range)
    classifier_SVM = GridSearchCV(svm_classifier, param_grid=param_grid, cv=cv, n_jobs=n_jobs)
    classifier_SVM.fit(X_train, y_train)
    best_C_SVM = classifier_SVM.best_estimator_.C
    best_gamma_SVM = classifier_SVM.best_estimator_.gamma

    estimator_SVM = SVC(C=best_C_SVM, gamma=best_gamma_SVM, random_state=0, probability=True)
    print "Best parameters are for SVM: C",best_C_SVM, "gamma", best_gamma_SVM, '\n'
    filename_svm = out_folder + "/SVM_all_features.pkl"
    joblib.dump(estimator_SVM, filename_svm, compress=9)

    #For RF
    #max_depth=np.linspace(1,20,5) #By default
    max_depth=np.linspace(1,10,5)
    n_estimators=np.arange(10, 300, 10) #n_estimator
    #estimatorR = grid_searchCV(clf2, cv, X_train, y_train)

    grid_search_random_forest = GridSearchCV(estimator=random_forest_classifier, cv=cv, param_grid=dict(n_estimators=n_estimators, max_depth=max_depth), n_jobs=n_jobs, scoring='f1')
    grid_search_random_forest.fit(X_train, y_train)
    print "classifier is", grid_search_random_forest
    best_max_depth_random_forest = grid_search_random_forest.best_estimator_.max_depth #bEST MAX_DEPTH
    best_n_estimators_random_forest = grid_search_random_forest.best_estimator_.n_estimators #Best n_estimator

    estimator_random_forest = RandomForestClassifier(n_estimators=best_n_estimators_random_forest, max_depth=best_max_depth_random_forest, random_state=0, n_jobs=n_jobs)
    estimator_random_forest.fit(X_train, y_train)
    print "Best parameters are for the first: max_depth =", best_max_depth_random_forest, "n_estimator:", best_n_estimators_random_forest, '\n'
    print "RF is", estimator_random_forest
    filename_random_forest = out_folder + "/RF_all_features.pkl"
    joblib.dump(estimator_random_forest, filename_random_forest, compress=9)
    print "Saved Random Forest Classifier....."

    grid_search_adaboost = GridSearchCV(estimator=adaboost_classifier, cv=cv, param_grid=dict(n_estimators=n_estimators, base_estimator__max_depth=max_depth), n_jobs=n_jobs, scoring='f1')
    grid_search_adaboost.fit(X_train, y_train)
    best_max_depth_adaboost = grid_search_adaboost.best_estimator_.base_estimator.max_depth #bEST MAX_DEPTH
    best_n_estimators_adaboost = grid_search_adaboost.best_estimator_.n_estimators #Best n_estimator
    print "Best parameters are for the first Ada: max_depth =", best_max_depth_adaboost, "n_estimator:", best_n_estimators_adaboost, '\n'


    estimator_adaboost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=best_max_depth_adaboost), algorithm="SAMME", n_estimators=best_n_estimators_adaboost)
    estimator_adaboost.fit(X_train, y_train)
    print "ADA is", estimator_adaboost
    filename_adaboost = out_folder + "/ADA_all_features.pkl"
    joblib.dump(estimator_adaboost, filename_adaboost, compress=9)
    print "Saved AdaBoost Classifier....."

    estimator_ensemble_classifier = EnsembleClassifier(clfs=[estimator_SVM, estimator_random_forest, estimator_adaboost], voting='hard')
    estimator_ensemble_classifier.fit(X_train, y_train)

    #prediction by ensemble classifier
    #y_pred = estimator_ensemble_classifier.predict(X_train)
    filename_ensemble_classifier = out_folder + "/Ensembled_Epigenome_Genome_model.pkl"
    joblib.dump(estimator_ensemble_classifier, filename_ensemble_classifier, compress=9)

    for clf, label in zip([estimator_SVM, estimator_random_forest, estimator_adaboost, estimator_ensemble_classifier], ['SVM', 'Random Forest', 'AdaBoost', 'Ensemble']):
        scores_accuracy = cross_validation.cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores_accuracy.mean(), scores_accuracy.std(), label))

    for clf, label in zip([estimator_SVM, estimator_random_forest, estimator_adaboost, estimator_ensemble_classifier], ['SVM', 'Random Forest', 'AdaBoost', 'Ensemble']):
        scores_f1 = cross_validation.cross_val_score(clf, X_train, y_train, cv=cv, scoring='f1')
        print("F1: %0.2f (+/- %0.2f) [%s]" % (scores_f1.mean(), scores_f1.std(), label))

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


def combine_rfs(rf_a, rf_b):
    rf_a.estimators_ += rf_b.estimators_
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
    n_test = round((indices.size*.20)/2) #No. of instances for each class #SHUFFLE DATA X and Y for splitting purpose
    n_test = int(n_test)
    print "The total no. of instances in straitified test samples are:", n_test ,"\n"
    test_indices=np.concatenate([(indices_1[:n_test]), (indices_0[:n_test])])
    np.random.shuffle(test_indices)
    train_indices=np.concatenate([(indices_1[n_test:]), (indices_0[n_test:])])
    np.random.shuffle(train_indices)
    X_test = Xtrain[test_indices]
    y_test = Ytrain[test_indices]
    X_train = Xtrain[train_indices]
    y_train = Ytrain[train_indices]
    print X_test.shape, y_test.shape, X_train.shape , y_train.shape
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    main(sys.argv[1:])
