import pandas as pd
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf

import warnings

warnings.filterwarnings('ignore')
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Read Data
def repair_data(raw_data, col, treshold):
    data = raw_data.copy(deep=True)
    ind_missing_greater = data[data[col] > treshold[1]].index
    ind_missing_less = data[data[col] < treshold[0]].index


    for index in ind_missing_greater:
        data[col][index] = data[col][index - 1] if index > 0 else data[col][index + 1]

    for index in ind_missing_less:
        data[col][index] = data[col][index - 1] if index > 0 else data[col][index + 1]
    return data



def prepare_data(data, time_delta = 3600):
    bytes = []
    packets = []
    timestamps = []
    time = data.Time[0] + time_delta
    sumB = 0
    sumP = 0
    for i in data.index:
        if data.Time[i] <= time:
            sumB += data.Bytes[i]
            sumP += data.Packets[i]
            if i == data.index[-1]:
                break
            elif data.Time[i + 1] > time:
                timestamps.append(time)
                bytes.append(sumB)
                packets.append(sumP)
                sumB = 0
                sumP = 0
                time += time_delta
    print('dsgfsdgdf', len(timestamps), len(bytes), len(packets))
    prom  = np.array([timestamps, bytes, packets]).transpose()
    res = pd.DataFrame(prom, columns=['Time', 'Bytes', 'Packets'])
    print(res.shape)
    return res

def make_sample(data, column, coef):
    data = data.to_numpy()
  #  print(data)
    dataset_in = np.empty((data.shape[0] - p, N + 1))
    dataset_out = np.empty(data.shape[0] - p)
    print(tab.shape, p)

    for i in range(tab.shape[0] - p):
        dataset_in[i, :] = np.array([data[i + p - k, column] for k in range(N, 0, -1)] + [data[i, column]])*coef[column]
        dataset_out[i] = data[i + p, column]*coef[column]
    return (dataset_in, dataset_out)


#Time interval for discretization
timeDelta = 60*60*1
#Periodic component of time series
periodSEC = 60*60*24
#Scaling coeeficient for time series
#scale_coeff = [1, 1e-9, 1e-3]
scale_coeff = [1, 1e-9, 1e-7]
#Number of previos timestamps for predicting
N = 6
#Part of sample dedicated for validation
#val = 0.3
if not os.path.exists('dataset/test_data_prepared.csv'):
    if not os.path.exists('dataset/test_data_repaired.csv'):
        print("Loading of raw test data")
        test_data = pd.read_csv('dataset/test_data.csv', sep=';').dropna()
        print(test_data.shape)
        test_data.columns = ['Type', 'Time', 'Bytes', 'Packets']
        test_data = repair_data(test_data, "Bytes", [100000, 1039828])
        test_data = repair_data(test_data, "Packets", [100, 2000])
        test_data.to_csv('dataset/test_data_repaired.csv', sep=';', index=False)
        print("Test data repaired")
    else:
        print("Loading of repaired test data")
        test_data = pd.read_csv('dataset/test_data_repaired.csv', sep=';').dropna()
    test_tab = prepare_data(test_data, timeDelta)
    test_tab.to_csv('dataset/test_data_prepared.csv', sep=';', index=False)
    print("Test data prepaired")
else:
    print("Loading of prepared data")
    test_tab = pd.read_csv('dataset/test_data_prepared.csv', sep=';').dropna()




if not os.path.exists('dataset/train_data_prepared.csv'):
    if not os.path.exists('dataset/train_data_repaired.csv'):
        print("Loading of raw data")
        train_data = pd.read_csv('dataset/train_data.csv', sep=';').dropna()
        print(train_data.shape)
        train_data.columns = ['Type', 'Time', 'Bytes', 'Packets']
        train_data = repair_data(train_data, "Bytes", [100000, 1039828])
        train_data = repair_data(train_data, "Packets", [100, 2000])
        train_data.to_csv('dataset/train_data_repaired.csv', sep=';', index=False)
        print('Train data repaired')
    else:
        print("Loading of repaired data")
        train_data = pd.read_csv('dataset/train_data_repaired.csv', sep=';').dropna()
    tab = prepare_data(train_data, timeDelta)
    tab.to_csv('dataset/train_data_prepared.csv', sep=';', index=False)
    print('Train data prepared')
else:
    print("Loading of prepared data")
    tab = pd.read_csv('dataset/train_data_prepared.csv', sep=';')

print('train_shape: ', tab.shape)
print('train: ', tab)
print('test_shape: ', test_tab.shape)
print('train: ', test_tab)


#ds = np.array(ff)
# избавляемся от первого непонятного всплеска
p = int(periodSEC/timeDelta)
'''
offset = 7
xdata = tab[offset+1:offset+p+1, 0]
ydata1 = tab[offset+1:offset+p+1, 1]
ydata2 = tab[offset+p+1:offset+2*p+1, 1]
ydata3 = tab[offset+2*p+1:offset+3*p+1, 1]
ydata4 = tab[offset+3*p+1:offset+4*p+1, 1]
ydata5 = tab[offset+4*p+1:offset+5*p+1, 1]
ydata6 = tab[offset + 5*p+1:offset+6*p+1, 1]
ydata7 = tab[offset + 6*p+1:offset+7*p+1, 1]
ydata8 = tab[offset + 7*p+1:offset+8*p+1, 1]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(xdata, ydata1, label="vt")
ax.plot(xdata, ydata2, label="sr")
ax.plot(xdata, ydata3, label="cht")
ax.plot(xdata, ydata4, label="ptn")
ax.plot(xdata, ydata5, label="sb")
ax.plot(xdata, ydata6, label="vskr")
ax.plot(xdata, ydata7, label="pn1")
ax.plot(xdata, ydata8, label="vt1")
ax.legend()

#ax.plot(xdata, ydata, color='tab:blue')
#ax.plot(xdata, zdata, color='tab:orange')
#ax.set_xlim([xdata[0], xdata[-1]])
#ax.set_ylim([1.2e9, 1.3e9])
# ax.set_ylim(train_dataset[:,1][1])
plt.show()
'''
predicting_parameter = 2
dataset_in, dataset_out = make_sample(tab, predicting_parameter, scale_coeff)
test_data_in, test_data_out = make_sample(test_tab, predicting_parameter, scale_coeff)

print(dataset_in.shape)
print(dataset_out.shape)
def compile_and_fit(model, trainX, trainY, val, patience=1000, MAX_EPOCHS=10000):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='auto')

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
#                  optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.05),
                  optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.005, epsilon=1e-4),
                  metrics=[tf.keras.metrics.MeanAbsolutePercentageError(), tf.keras.metrics.RootMeanSquaredError()])

    history = model.fit(trainX, trainY, batch_size=20, epochs=MAX_EPOCHS,
                        validation_split=val,
                        #verbose=True,
                        callbacks=[early_stopping])
    return history
multi_step_dense = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(N+1,)),
    tf.keras.layers.Dense(units=50, activation='tanh'),
    tf.keras.layers.Dense(units=1, activation='relu')
])
history = compile_and_fit(multi_step_dense, dataset_in, dataset_out, 0.2, MAX_EPOCHS=10000)
predict_out = multi_step_dense.predict(test_data_in)
fig = plt.figure()
MAPE = tf.keras.metrics.MeanAbsolutePercentageError()
MAPE.update_state(test_data_out, predict_out)
RMSE = tf.keras.metrics.RootMeanSquaredError()
RMSE.update_state(test_data_out, predict_out)
print('MAPE:', MAPE.result().numpy(), 'RMSE:', RMSE.result().numpy() )
ax = fig.add_subplot(1, 1, 1)
xses = range(test_data_out.shape[0])
ax.plot(xses, test_data_out, label="1")
ax.plot(xses, predict_out, label="2")
ax.legend()




plt.show()
if (MAPE.result().numpy() <=30):
    multi_step_dense.save('models/model_'+str(predicting_parameter) + '_MAPE-' + str(MAPE.result().numpy()) + '_RMSE-' + str(RMSE.result().numpy()) + '.h5')
    plt.savefig('pics/' + str(predicting_parameter) + '_MAPE-' + str(MAPE.result().numpy()) + '_RMSE-' + str(RMSE.result().numpy()) + '.png')


'''

# predicted_un = predicted_un * -1

def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100

print("MAPE BYTES = ", mean_absolute_percentage_error(y_test.values[:, 0], predicted.values[:, 0]))
print("MAPE PACKAGES = ", mean_absolute_percentage_error(y_test.values[:, 1], predicted.values[:, 1]))


multi_step_dense = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dense(units=2)
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    #     tf.keras.layers.Reshape([1, -1]),
])

history = compile_and_fit(multi_step_dense, X_train, y_train, (X_val, y_val), MAX_EPOCHS=100000)

# %% md

# test on trained data

# %%

predicted = multi_step_dense.predict(X_val)

f, ax = plt.subplots(1)
# xdata = y_test.values[:,0]
ax.plot(y_val[:, 0])
ax.plot(predicted[:, 0])
# ax.set_ylim(train_dataset[:,2][1])
plt.show(f)

# %%

predicted = multi_step_dense.predict(X_train)

f, ax = plt.subplots(1)
# xdata = y_test.values[:,0]
ax.plot(y_train[:, 0])
ax.plot(predicted[:, 0])
# ax.set_ylim(train_dataset[:,2][1])
plt.show(f)

# %%

predicted = multi_step_dense.predict(X_test)
multi_step_dense.evaluate(X_test, y_test)

# %%


# %% md

# Predict Test Dataset


# %%

predicted = multi_step_dense.predict(X_test)
multi_step_dense.evaluate(X_test, y_test)

# %%

predicted = pd.DataFrame(predicted, columns=['Bytes', 'Packages'])
y_test = pd.DataFrame(y_test, columns=['Bytes', 'Packages'])

# y_test_un = unNormalizedData(y_test,trainMean, trainSTD)
# predicted_un = unNormalizedData(predicted, trainMean, trainSTD)
# predicted_un

# %%

plt.plot(y_test.values[:, 0])

# %%

plt.plot((predicted).values[:, 0])

# %%

f, ax = plt.subplots(1)
xdata = y_test.values[:, 0]
ax.plot(xdata)
ax.plot(predicted.values[:, 0])
# ax.set_ylim(train_dataset[:,2][1])
plt.show(f)


# %%

def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100


# predicted_un = predicted_un * -1
print("MAPE BYTES = ", mean_absolute_percentage_error(y_test.values[:, 0], predicted.values[:, 0]))
print("MAPE PACKAGES = ", mean_absolute_percentage_error(y_test.values[:, 1], predicted.values[:, 1]))

# %%

inversed_predicted = abs_scaler.inverse_transform(predicted)
inverse_y_test = abs_scaler.inverse_transform(y_test)

print("Inversed MAPE BYTES = ", mean_absolute_percentage_error(inverse_y_test[:, 0], inversed_predicted[:, 0]))
print("Inversed MAPE PACKAGES = ", mean_absolute_percentage_error(inverse_y_test[:, 1], inversed_predicted[:, 1]))

# %%

f, ax = plt.subplots(1)
ax.plot(inverse_y_test[:, 0])
ax.plot(inversed_predicted[:, 0])
# ax.set_ylim(train_dataset[:,2][1])
plt.show(f)

# %% md

# Save Model

# %%

multi_step_dense.save("ouput.h5")
'''
# %%


# %%


# %%


# %%


# %%


# %%


# %%


