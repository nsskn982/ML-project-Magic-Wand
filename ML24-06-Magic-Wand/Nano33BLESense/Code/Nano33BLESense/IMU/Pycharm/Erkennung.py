import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

def read_data(file_path):
    column_names = ['user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
    data = pd.read_csv(file_path, header=None, names=column_names)
    return data

def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma

def data_classify(new_data, act):
    n = 0
    m = 0
    for x in data['activity']:
        if x == act:
            new_data.loc[n] = data.loc[m]
            n = n+1
        m += 1

def to_array(data):
    segments = np.zeros((1, 3))
    labels = np.zeros((1))
    c = 0
    for i in data.index:
        segments = np.vstack([segments, np.array(
            [data.loc[c, 'x-axis'], data.loc[c, 'y-axis'],
             data.loc[c, 'z-axis']])])
        # labels = np.append(labels, data["activity"].loc[count])
        labels = np.vstack([labels, np.array([data["activity"].loc[c]])])
        c += 1
    labels = np.delete(labels, 0, 0).astype(int)
    segments = np.delete(segments, 0, 0)
    return segments, labels

data = read_data('C:/Users/xx412/PycharmProjects/pythonProject/WISDM.txt')
data = data.dropna(axis=0, how='any')

i = 0
for x in data['x-axis']:
    if x == 0:
        data = data.drop(labels=i, axis=0)
    i = i + 1
data.reset_index(drop=True, inplace=True)

data['activity'] = data['activity'].replace('Walking', '0')
data['activity'] = data['activity'].replace('Jogging', '1')
data['activity'] = data['activity'].replace('Upstairs', '2')
data['activity'] = data['activity'].replace('Downstairs', '3')
data['activity'] = data['activity'].replace('Sitting', '4')
data['activity'] = data['activity'].replace('Standing', '5')

data['activity'] = data['activity'].astype(np.float)

"归一化数据"
data['x-axis'] = feature_normalize(data['x-axis'])
data['y-axis'] = feature_normalize(data['y-axis'])
data['z-axis'] = feature_normalize(data['z-axis'])
print(data.shape)
data.to_csv('data.txt')

data0 = pd.DataFrame(columns=('user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis'))
data1 = pd.DataFrame(columns=('user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis'))
data2 = pd.DataFrame(columns=('user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis'))
data3 = pd.DataFrame(columns=('user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis'))
data4 = pd.DataFrame(columns=('user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis'))
data5 = pd.DataFrame(columns=('user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis'))

data_classify(data0, 0)
data_classify(data1, 1)
data_classify(data2, 2)
data_classify(data3, 3)
data_classify(data4, 4)
data_classify(data5, 5)

segments0, labels0 = to_array(data0)
segments1, labels1 = to_array(data1)
segments2, labels2 = to_array(data2)
segments3, labels3 = to_array(data3)
segments4, labels4 = to_array(data4)
segments5, labels5 = to_array(data5)

def train_data(train):
    i = int(train.shape[0]*0.6)
    return  train[:i]
def test_data(test):
    i = int(test.shape[0] * 0.6)
    return test[i+1:]

x_train = np.vstack([train_data(segments0), train_data(segments1), train_data(segments2),
                     train_data(segments3), train_data(segments4), train_data(segments5)])
x_test = np.vstack([test_data(segments0), test_data(segments1), test_data(segments2),
                     test_data(segments3), test_data(segments4), test_data(segments5)])
y_train = np.vstack([train_data(labels0), train_data(labels1), train_data(labels2),
                     train_data(labels3), train_data(labels4), train_data(labels5)])
y_test = np.vstack([test_data(labels0), test_data(labels1), test_data(labels2),
                     test_data(labels3), test_data(labels4), test_data(labels5)])


model = tf.keras.Sequential()
model.add(layers.Dense(data.shape[1], activation='relu', input_shape=(data.shape[1],)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(np.unique(y_train).size, activation='softmax'))
y_train = tf.keras.utils.to_categorical(y_train, 6)
y_test = tf.keras.utils.to_categorical(y_test, 6)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train,
                    y_train,
                    batch_size=32,
                    epochs=1200,
                    validation_data=(x_test, y_test),
                    verbose=1)

eval_result = model.evaluate(x_test, y_test)
predictions = model.predict(x_test)
test_score = model.evaluate(x_test, y_test)


accuracy = history.history['accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']
epochs = np.arange(len(accuracy)) + 1

plt.subplot(211)
plt.title(f'Test accuracy: {round(test_score[1], 3)}')
plt.plot(epochs, accuracy, label='Accuracy')
plt.plot(epochs, val_accuracy, label='Validate accuracy')
plt.grid(True)
plt.legend()

plt.subplot(212)
plt.title(f'Test loss: {round(test_score[0], 3)}')
plt.grid(True)

print('Loss:', eval_result[0])
print('Accuracy:', eval_result[1]*100)

model.save('./new.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()

print(tflite_model)
open('model.tflite', 'wb').write(tflite_model)

def hex_to_c_arrary(hex_data, var_name):
    c_str = ''
    c_str += '#ifndef ' + var_name.upper() + '_H\n'
    c_str += '#define ' + var_name.upper() + '_H\n\n'

    c_str += '\nunsigned int' + var_name + '_len=' +str(len(hex_data)) + ';\n'

    c_str += 'const unsigned char ' + var_name + '[] = {'
    hex_array = []
    for i, val in enumerate(hex_data):
        hex_str = format(val, '#04x')

        if(i+1) < len(hex_data):
            hex_str += ','
        if(i+1) % 12 == 0:
            hex_str += '\n'
        hex_array.append(hex_str)

    c_str += '\n' + format(' '.join(hex_array)) + '\n};\n\n'

    c_str += '#endif //' + var_name.upper() + '_H'

    return c_str
c_modek_name = 'model'
with open(c_modek_name + '.h', 'w') as file:
    file.write(hex_to_c_arrary(tflite_model, c_modek_name))