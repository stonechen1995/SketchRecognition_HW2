# Load data
import pandas as pd

df = pd.read_csv('extended_features.csv')
_, n = df.shape
X = df.iloc[:, range(n)]
y = df.iloc[:, n]

# shuffle data
from sklearn.utils import shuffle
df = shuffle(df, random_state=0)

# create 3 different classifiers
from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X, y)

from sklearn.tree import DecisionTreeClassifier
decisionTree_clf = DecisionTreeClassifier()
decisionTree_clf.fit(X, y)

from sklearn.ensemble import RandomForestClassifier
randomForest_clf = RandomForestClassifier()
randomForest_clf.fit(X, y)

clf_dict = {"dummy_clf": dummy_clf, "decisionTree_clf": decisionTree_clf,
            "randomForest_clf": randomForest_clf}

import pickle
for clf in clf_dict:
    path = f'./{clf}.sav'
    pickle.dump(clf_dict[clf], open(path, "wb"))