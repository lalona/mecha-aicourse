# sepal-width, sepal-length, petal-width and petal-length.
# https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Assign colum names to the dataset
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
irisdata = pd.read_csv('iris.data', names=colnames)

X = irisdata.drop('Class', axis=1)
y = irisdata['Class']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# polynomial, Gaussian, and sigmoid kernels
svclassifier = SVC(kernel='poly', degree=8)
# svclassifier = SVC(kernel='rbf')
# svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
