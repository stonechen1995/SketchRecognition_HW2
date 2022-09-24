# Load data
import pandas as pd

df = pd.read_csv('features.csv')
X = df.iloc[:, range(13)] # classes
y = df.iloc[:, 13] # label

# shuffle data
from sklearn.utils import shuffle
df = shuffle(df, random_state=0)

# create 7 different classifiers
from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X, y)

from sklearn.tree import DecisionTreeClassifier
decisionTree_clf = DecisionTreeClassifier()
decisionTree_clf.fit(X, y)

from sklearn.ensemble import RandomForestClassifier
randomForest_clf = RandomForestClassifier()
randomForest_clf.fit(X, y)

from sklearn.neural_network import MLPClassifier
mlp_clf = MLPClassifier(max_iter=2000)
mlp_clf.fit(X, y) # will cause warning

from sklearn.naive_bayes import GaussianNB
gaussian_clf = GaussianNB()
gaussian_clf.fit(X, y)

from sklearn.neighbors import KNeighborsClassifier
kN_clf = KNeighborsClassifier()
kN_clf.fit(X, y)

from sklearn.tree import ExtraTreeClassifier
extraTree_clf = ExtraTreeClassifier()
extraTree_clf.fit(X, y)

clf_dict = {"dummy_clf": dummy_clf, "decisionTree_clf": decisionTree_clf,
            "randomForest_clf": randomForest_clf, "mlp_clf": mlp_clf,
            "gaussian_clf": gaussian_clf, "kN_clf": kN_clf,
            "extraTree_clf": extraTree_clf}

import pickle
for clf in clf_dict:
    path = f'./7PickledCLF/{clf}.pkl'
    pickle.dump(clf_dict[clf], open(path, "wb"))
    