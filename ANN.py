# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf

# Importing dataset
dataset = pd.read_csv("D:\\EDUCATION\\OTHERS\\MACHINE LEARNING\\Forest cover prediction\\covtype.csv")

# Encoding dependent variable
X = dataset.iloc[:, :55]
scaler = StandardScaler()
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(dataset["Cover_Type"])

# Train-Test splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Normalization
train_norm = X_train.iloc[:, :10]
test_norm = X_test.iloc[:, :10]

std_scale = StandardScaler().fit(train_norm)

X_train_norm = std_scale.transform(train_norm)
training_norm_col = pd.DataFrame(
    X_train_norm, index=train_norm.index, columns=train_norm.columns)
X_train.update(training_norm_col)

X_test_norm = std_scale.transform(test_norm)
testing_norm_col = pd.DataFrame(
    X_test_norm, index=test_norm.index, columns=test_norm.columns)
X_test.update(testing_norm_col)

# Creating ANN
cover_model = tf.keras.Sequential()

# Adding 1st layer
cover_model.add(tf.keras.layers.Dense(
    units=64, activation='relu', input_shape=(X_train.shape[1],)))
cover_model.add(tf.keras.layers.Dense(units=64, activation='relu'))
cover_model.add(tf.keras.layers.Dense(units=8, activation='softmax'))

# Adding output layer
cover_model.compile(optimizer=tf.optimizers.Adam(),
                    loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_cover = cover_model.fit(
    X_train, y_train, epochs=8, batch_size=64, validation_data=(X_test, y_test))

# Plotting accuracy graph
plt.plot(history_cover.history['accuracy'])
plt.plot(history_cover.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
