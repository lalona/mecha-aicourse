import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

def create_model():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(7),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    return model



titanic_data = pd.read_csv("data/train.csv")



# La cabina se quito porque faltan muchos datos
X = titanic_data.drop(['Survived', "Name", "Ticket", "Cabin", "Embarked"], axis=1)
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

print(X_train)

model = create_model()

target = X_train
dataset = tf.data.Dataset.from_tensor_slices((X_train.values, target.values))

train_dataset = dataset.shuffle(len(X_train)).batch(1)

model.fit(train_dataset, y_train, epochs=5)

model.evaluate(X_test,  y_test, verbose=1)