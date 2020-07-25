import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

titanic_data = pd.read_csv("data/train.csv")



# La cabina se quito porque faltan muchos datos
X = titanic_data.drop(['Survived', "Name", "Ticket", "Cabin", "Embarked", "SibSp", "Parch"], axis=1)
y = titanic_data['Survived']

# Get list of categorical variables
s = (X.dtypes == 'object')
object_cols = list(s[s].index)

print(object_cols)

label_encoder = LabelEncoder()

X['Sex'] = label_encoder.fit_transform(X['Sex'])

my_imputer = SimpleImputer()

X = pd.DataFrame(my_imputer.fit_transform(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#X_train = preprocessing.scale(X_train)

print(X_train)

svclassifier = SVC(kernel='poly', degree=4)
#svclassifier = SVC(kernel='rbf')
#svclassifier = SVC(kernel='sigmoid')

svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))