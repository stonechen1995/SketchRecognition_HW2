# load data
import pandas as pd

df = pd.read_csv('shape_features.csv')
_, n = df.shape
X = df.iloc[:, range(n - 1)]
y = df.iloc[:, n - 1]


# shuffle data
from sklearn.utils import shuffle
df = shuffle(df, random_state=0)

# feature selection
from sklearn.feature_selection import VarianceThreshold, SelectFromModel

sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform(X)

# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# X_new = SelectKBest(chi2, k=2).fit_transform(X, y)


# use train/test split
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # 80/20 split in this case

# evaluate with metrics
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

# from sklearn.ensemble import ExtraTreesClassifier
# extraTree_clf = ExtraTreesClassifier(n_estimators=50)
# extraTree_clf = extraTree_clf.fit(X_train, y_train)
# print(extraTree_clf.feature_importances_)
# model = SelectFromModel(extraTree_clf, prefit=True)
# X_train = model.transform(X)

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
# randomForest_clf = Pipeline([
#   ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
#   ('classification', RandomForestClassifier())
# ])
randomForest_clf = RandomForestClassifier()
randomForest_clf.fit(X, y)
# y_pred = randomForest_clf.predict(X_test)
# f1_score(y_test, y_pred, average='weighted')  # must have averaging method for multi-class
# print(classification_report(y_test, y_pred))
print(randomForest_clf.feature_importances_)  # give weights for the features by column index
print(randomForest_clf.feature_names_in_)

# or cross validation
# from sklearn.model_selection import cross_validate
# cross_validate(randomForest_clf, X, y, scoring='f1_micro', cv=10, return_train_score=True)  # you can pass custom scoring methods

# confusion matrix
# from sklearn.metrics import confusion_matrix
# confusion_matrix(y_test, y_pred)

# from sklearn.metrics import ConfusionMatrixDisplay
# import matplotlib.pyplot as plt
# matrix = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
# matrix.plot(cmap=plt.cm.Blues)
# plt.title('Confusion Matrix')
# plt.show()

import pickle
pickle.dump(randomForest_clf, open('RandomForestClassifier.sav', "wb"))
