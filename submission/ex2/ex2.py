# Load data
import pandas as pd

df = pd.read_csv('features.csv')
X = df.iloc[:, range(13)]  # classes
y = df.iloc[:, 13]  # label
print(f'X.shape = {X.shape}')

# shuffle data
from sklearn.utils import shuffle
df = shuffle(df, random_state=0)

# feature selection
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X_new = sel.fit_transform(X)
print(f'X_new.shape = {X_new.shape}')

# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# X_new = SelectKBest(chi2, k=2).fit_transform(X, y)

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
mlp_clf.fit(X, y)  # will cause warning

from sklearn.naive_bayes import GaussianNB
gaussian_clf = GaussianNB()
gaussian_clf.fit(X, y)

from sklearn.neighbors import KNeighborsClassifier
kN_clf = KNeighborsClassifier()
kN_clf.fit(X, y)

from sklearn.feature_selection import SelectFromModel
from sklearn.tree import ExtraTreeClassifier
extraTree_clf = ExtraTreeClassifier()
extraTree_clf.fit(X, y)
print(f'feature_importances_ = {sum(extraTree_clf.feature_importances_)}')
extraTree_model = SelectFromModel(extraTree_clf, prefit=True)
X_new = extraTree_model.transform(X)
print(f'X_new.shape: {X_new.shape}')

clf_dict = {"DummyClassifier": dummy_clf, "DecisionTreeClassifier": decisionTree_clf,
            "RandomForestClassifier": randomForest_clf, "MLPClassifier": mlp_clf,
            "GaussianNB": gaussian_clf, "KNClassifier": kN_clf,
            "ExtraTreeClassifier": extraTree_clf}

import pickle
for clf in clf_dict:
    path = f'./{clf}.sav'
    pickle.dump(clf_dict[clf], open(path, "wb"))
    