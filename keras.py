import keras
import numpy as np
import pandas as pd
from time import time
from tensorflow import config
from sklearn.preprocessing import LabelBinarizer

train_url = 'https://raw.githubusercontent.com/Aryanchaturvedi075/Sign_Language_MNIST/main/sign_mnist_train.csv'
test_url = 'https://raw.githubusercontent.com/Aryanchaturvedi075/Sign_Language_MNIST/main/sign_mnist_test.csv'
train_df = pd.read_csv(train_url)
test_df = pd.read_csv(test_url)

# For Training Data
train_label = train_df['label']
trainset = train_df.drop(['label'],axis=1)

# For Testing Data
test_label = test_df['label']
testset = test_df.drop(['label'],axis=1)

X_train = trainset.values.reshape(-1,28,28,1).astype(np.float64)
X_test = testset.values.reshape(-1,28,28,1).astype(np.float64)

X_train /= 255
X_test /= 255

X_train -= np.mean(X_train, axis = 0)
X_test -= np.mean(X_test, axis = 0)

lb=LabelBinarizer()
Y_train = lb.fit_transform(train_label)
Y_test = lb.fit_transform(test_label)

model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(units=512,activation='relu'),
        keras.layers.Dropout(rate=0.001),
        keras.layers.Dense(units=256,activation='relu'),
        keras.layers.Dropout(rate=0.001),
        keras.layers.Dense(units=24,activation='softmax')
    ]
)

# Key Changes made here
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),   # Ideal Learning Rate
    loss="categorical_crossentropy",
    metrics=["accuracy"])

time_start = time()
model.fit(
    X_train,
    Y_train,
    batch_size = 128,                                       # Highest Accuracy Batch Size
    epochs = 100,
    validation_data=(X_test, Y_test),
    shuffle= False
)
print("time spent training {:0.3f}".format(time() - time_start))
test_loss, test_acc = model.evaluate(X_test, Y_test)
print("Test accuracy:", test_acc * 100, "\n")