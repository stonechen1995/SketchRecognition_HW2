# Load data
import pandas as pd

df = pd.read_csv('gesture_features.csv')
_, n = df.shape
X = df.iloc[:, range(n - 1)]
y = df.iloc[:, n - 1]

# shuffle data
from sklearn.utils import shuffle
df = shuffle(df, random_state=0)

# feature selection
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform(X)

# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# X_new = SelectKBest(chi2, k=2).fit_transform(X, y)

# use train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # 80/20 split in this case

# evaluate with metrics
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

# create 3 different classifiers
from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
y_pred = dummy_clf.predict(X_test)
f1_score(y_test, y_pred, average='weighted')  # must have averaging method for multi-class
print(classification_report(y_test, y_pred))


from sklearn.tree import DecisionTreeClassifier
decisionTree_clf = DecisionTreeClassifier()
decisionTree_clf.fit(X_train, y_train)
y_pred = decisionTree_clf.predict(X_test)
f1_score(y_test, y_pred, average='weighted')  # must have averaging method for multi-class
print(classification_report(y_test, y_pred))


from sklearn.ensemble import RandomForestClassifier
randomForest_clf = RandomForestClassifier()
randomForest_clf.fit(X_train, y_train)
y_pred = randomForest_clf.predict(X_test)
f1_score(y_test, y_pred, average='weighted')  # must have averaging method for multi-class
print(classification_report(y_test, y_pred))


clf_dict = {"DummyClassifier": dummy_clf, "DecisionTreeClassifier": decisionTree_clf,
            "RandomForestClassifier": randomForest_clf}

import pickle
for clf in clf_dict:
    path = f'./{clf}.sav'
    pickle.dump(clf_dict[clf], open(path, "wb"))
