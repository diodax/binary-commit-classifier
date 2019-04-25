import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


class BinaryClassifier(object):
    """
    The class that contains the methods related to the Scikit-multilearn classifier.
    """

    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train.astype(float)
        self.x_test = x_test.astype(float)

        self.y_train = y_train.astype(bool)
        self.y_test = y_test.astype(bool)
        self.classifier = None

    def apply_gaussian_nb(self):
        self.classifier = GaussianNB()
        self.classifier.fit(self.x_train, self.y_train)
        predictions = self.classifier.predict(self.x_test)
        a = accuracy_score(self.y_test, predictions)
        f = f1_score(self.y_test, predictions, average='micro')
        return a, f

    def apply_logistic_regression(self):
        self.classifier = LogisticRegression(solver='lbfgs')
        self.classifier.fit(self.x_train, self.y_train)
        predictions = self.classifier.predict(self.x_test)
        a = accuracy_score(self.y_test, predictions)
        f = f1_score(self.y_test, predictions, average='micro')
        return a, f

    def apply_random_forest(self):
        self.classifier = RandomForestClassifier(n_estimators=10)
        self.classifier.fit(self.x_train, self.y_train)
        predictions = self.classifier.predict(self.x_test)
        a = accuracy_score(self.y_test, predictions)
        f = f1_score(self.y_test, predictions, average='micro')
        return a, f

    def apply_decision_tree(self, max_depth=3):
        self.classifier = tree.DecisionTreeClassifier(max_depth=max_depth)
        self.classifier.fit(self.x_train, self.y_train)
        predictions = self.classifier.predict(self.x_test)
        a = accuracy_score(self.y_test, predictions)
        f = f1_score(self.y_test, predictions, average='micro')
        return a, f

    def apply_svm(self):
        self.classifier = SVC(probability=True, gamma='scale')
        self.classifier.fit(self.x_train, self.y_train)
        predictions = self.classifier.predict(self.x_test)
        a = accuracy_score(self.y_test, predictions)
        f = f1_score(self.y_test, predictions, average='micro')
        return a, f

    def apply_knn(self, k=3):
        self.classifier = KNeighborsClassifier(n_neighbors=k)
        self.classifier.fit(self.x_train, self.y_train)
        predictions = self.classifier.predict(self.x_test)
        a = accuracy_score(self.y_test, predictions)
        f = f1_score(self.y_test, predictions, average='micro')
        return a, f
