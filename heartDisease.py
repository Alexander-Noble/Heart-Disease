import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


path = "/path/to/data.csv"
df = pd.read_csv(path)
data = df.iloc[:, 0:13]
data = data.values
data[data == "?"] = 0
labels = df.iloc[:, -1]
labels = labels.values
labels = np.divide(labels, 4)
labels[labels > 0.2] = 1



tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)


model = Sequential()

model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=['acc'])

history = model.fit(data, labels, epochs=64, validation_split=0.05, callbacks=[tbCallBack])


history_dict = history.history
loss_val = history_dict['loss']
acc_val = history_dict['acc']
val_loss_values = history_dict['val_loss']

epochs = range(1, 65)

plt.plot(epochs, loss_val, 'bo', label="Training loss")
#plt.plot(epochs, val_loss_values, 'b', label="Validation Loss")
plt.title("Training loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend

plt.show()
