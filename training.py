import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.layers import Dense, Activation, Dropout 
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow.python.keras as tf_keras
from keras import __version__
tf_keras.__version__ = __version__

from tensorflow.python.keras.engine import data_adapter


def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)

data_adapter._is_distributed_dataset = _is_distributed_dataset

df = pd.read_pickle("final_data/final_data.csv")
X = df["feature"].values
X = np.concatenate(X, axis=0).reshape(len(X), 40)

y=np.array(df["class_label"].tolist())
y=to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

model = Sequential([
    Dense(256, input_shape=X_train[0].shape),
    Activation("relu"),
    Dropout(0.5),
    Dense(256),
    Activation("relu"),
    Dropout(0.5),
    Dense(2, activation="softmax")

])

model.summary()

model.compile(
    loss = "categorical_crossentropy",
    optimizer="adam",
    metrics = ["accuracy"]
)

print("Model Score: \n")
history = model.fit(X_train, y_train, epochs=1000)
model.save("saved_model/model1.h5")
score = model.evaluate(X_test, y_test)
print(score)

print("Model classification report:\n")
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
print(classification_report(np.argmax(y_test, axis=1), y_pred))


