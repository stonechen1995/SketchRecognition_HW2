# load data
import pandas as pd

df = pd.read_csv('shape_features.csv')
_, n = df.shape
X = df.iloc[:, range(n - 1)]
y = df.iloc[:, n - 1]
# print(X.iloc[0, :])
X_new = df.iloc[:, [8, 10, 13, 16, 19, 29, 32, 34, 43]]
# print(X_new.shape)

# shuffle data
from sklearn.utils import shuffle
df = shuffle(df, random_state=0)

# feature selection
# from sklearn.feature_selection import VarianceThreshold, SelectFromModel
# sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
# X_new = sel.fit_transform(X)
# print(X_new.shape)
# print(X_new.iloc[0, :])

# from sklearn.ensemble import ExtraTreesClassifier
# extraTree_clf = ExtraTreesClassifier(n_estimators=50)
# extraTree_clf = extraTree_clf.fit(X_new, y)
# model = SelectFromModel(extraTree_clf, prefit=True)
# X_new = model.transform(X_new)
# print(X_new.shape)

# from sklearn.feature_selection import SelectFromModel
# from sklearn.linear_model import LogisticRegression
# # Selecting the Best important features according to Logistic Regression using SelectFromModel
# sfm_selector = SelectFromModel(estimator=LogisticRegression())
# sfm_selector.fit(X_new, y)
# X_new = sfm_selector.transform(X_new)
# print(X_new.shape)

# use train/test split
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2)  # 80/20 split in this case

# evaluate with metrics
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier
randomForest_clf = RandomForestClassifier()
randomForest_clf.fit(X_new, y)
# y_pred = randomForest_clf.predict(X_test)
# f1_score(y_test, y_pred, average='weighted')  # must have averaging method for multi-class
# print(classification_report(y_test, y_pred))
# print(randomForest_clf.feature_importances_)  # give weights for the features by column index
print(randomForest_clf.feature_names_in_)

# or cross validation
# from sklearn.model_selection import cross_validate
# score = cross_validate(randomForest_clf, X_new, y, scoring='f1_micro', cv=10, return_train_score=True)  # you can pass custom scoring methods
# print(score)

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
