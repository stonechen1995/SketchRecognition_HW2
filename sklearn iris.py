# Load data
import pandas as pd

df = pd.read_csv('iris.csv')

from sklearn.utils import shuffle
df = shuffle(df, random_state=0)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
x = df.iloc[:, [0, 1, 2, 3]]
y = df.iloc[:, 4]
clf.fit(x,y)


y = df['variety'] # by convention, y values are class labels
X = df.loc[:, df.columns != 'variety'] # x values are everything else (the features)


# Create classifiers
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()


# use train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2) # 80/20 split in this case
rf.fit(X_train,y_train) # .fit() works for all classifiers

# or cross validation
from sklearn.model_selection import cross_validate
cross_validate(rf, X, y, scoring='f1_micro',cv=10, return_train_score=True) # you can pass custom scoring methods


# evaluate with metrics
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

# generate predictions to use for metrics and confusion matrix
y_pred = rf.predict(X_test)
f1_score(y_test,y_pred,average='weighted') # must have averaging method for multi-class
print(classification_report(y_test,y_pred))

# confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
matrix = ConfusionMatrixDisplay(confusion_matrix(y_test,y_pred))
matrix.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()


# feature selection (many methods available, this uses a model to select)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X, y) # uses whole dataset, like the Weka default
clf.feature_importances_  # gives weights for the features by column index