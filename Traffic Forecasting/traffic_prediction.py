import pandas as pd
import numpy as np


import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')
train_data = pd.read_csv('dataset/train_data.csv', sep=';').dropna()
test_data = pd.read_csv('dataset/test_data.csv', sep=';').dropna()

train_data.columns = ['1','Time', 'Bytes', 'Packages']
test_data.columns = ['1','Time', 'Bytes', 'Packages']
print('train_shape: ',train_data.shape)
print('test_shape: ',test_data.shape)
train_data.values[0].copy()


def generateDataSetWithTimeDelta(dataset):  # Вместо времени используем delta Time
    dataset = dataset.values
    first_elemnt = dataset[0].copy()
    first_elemnt[1] = 3
    newDataset = [list(first_elemnt)]
    for index in range(1, len(dataset)):
        newDelta = dataset[index][1] - dataset[index - 1][1]

        next_element = list(dataset[index])
        next_element[1] = newDelta

        newDataset.append(next_element)

    return newDataset

withDeltaTimeTrainDataset = pd.DataFrame(generateDataSetWithTimeDelta(train_data), columns=['1','Delta', 'Bytes', 'Packages'])
withDeltaTimeTestDataset = pd.DataFrame(generateDataSetWithTimeDelta(test_data), columns=['1','Delta', 'Bytes', 'Packages'])

# Проверка разброса данных дельта времени
withDeltaTimeTrainDataset.boxplot(column=['Delta'])

# Проверка разброса данных байтов
withDeltaTimeTrainDataset.boxplot(column=['Bytes'])


# Отрбрасываем выбросы данных
def clearData(dataset):
    ind_missing = dataset[dataset['Bytes'] > 200000000].index
    dataset = dataset.drop(ind_missing, axis=0)

    ind_missing = dataset[dataset['Delta'] > 4000].index
    dataset = dataset.drop(ind_missing, axis=0)
    dataset.boxplot(column=['Bytes'])
    dataset.boxplot(column=['Delta'])
    return dataset




withoutEmissionsTrainData = clearData(withDeltaTimeTrainDataset).drop(columns=["1", "Delta"])
withoutEmissionsTestData = clearData(withDeltaTimeTestDataset).drop(columns=["1", "Delta"])

train_dataset = withoutEmissionsTrainData.values
n = len(train_dataset)
train_data = pd.DataFrame(train_dataset[0:int(n*0.7)], columns=[ 'Bytes', 'Packages'])
val_data = pd.DataFrame(train_dataset[int(n*0.7):], columns=[ 'Bytes', 'Packages'])


# Нормализовать данные
def normalizeData(data, train_mean, train_std):
    return (data - train_mean) / train_std


# Вернуть данные в обратном
def unNormalizedData(data, train_mean, train_std):
    return (data * train_std) + train_mean


# trainDeltaColumn = train_data["Delta"]
# valDeltaColumn = val_data["Delta"]

# trainMean = train_data.drop(columns=["Delta"]).mean()
# trainSTD = train_data.drop(columns=["Delta"]).std()

# normalizedTrain = normalizeData(train_data.drop(columns=["Delta"]), trainMean, trainSTD)
# normalizedTrain["Delta"] = trainDeltaColumn

# normalizedVal = normalizeData(val_data.drop(columns=["Delta"]), trainMean, trainSTD)
# normalizedVal["Delta"] = valDeltaColumn

trainMean = train_data.mean()
trainSTD = train_data.std()

normalizedTrain = normalizeData(train_data, trainMean, trainSTD)
normalizedVal = normalizeData(val_data, trainMean, trainSTD)


# %% md

# Generate Dataset

# %%

def generateDatasetWithSum(dataset, LAST_COUNT=5, DATA_PARAMETRS_COUNT=10):
    # LAST_COUNT - Количество элементов предшествующих предсказанному значению - [[Sum[0...5],Sum[6...10], ..., Sum[n...n+5]]
    # DATA_PARAMETRS_COUNT = 10 # Количество элементов в массиве подсуммы - Sum[0...5]
    # Сумма предыдущих должна соответсвовать сумме следующих
    # x  = [[Sum[0...5],Sum[6...10], ..., Sum[n...n+5]]
    # y = Sum[n+6...n+10]
    x_train = []
    y_train = []

    size = len(dataset)
    for index in range(0, size, DATA_PARAMETRS_COUNT):
        temp = []
        for nextIndex in range(0, LAST_COUNT):
            temp.append(list(dataset[
                             index + nextIndex * DATA_PARAMETRS_COUNT: index + nextIndex * DATA_PARAMETRS_COUNT + DATA_PARAMETRS_COUNT].sum(
                axis=0)))
        x_train.append(temp)
        y_train.append(list(dataset[
                            index + LAST_COUNT * DATA_PARAMETRS_COUNT: index + LAST_COUNT * DATA_PARAMETRS_COUNT + DATA_PARAMETRS_COUNT].sum(
            axis=0)))
    return (x_train, y_train)


# %%

X_train, y_train = generateDatasetWithSum(normalizedTrain)
X_val, y_val = generateDatasetWithSum(normalizedVal)


# %%


# %% md

# Train Model

# %%

def compile_and_fit(model, trainX, trainY, validData, patience=10, MAX_EPOCHS=10):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='auto')

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.SGD(learning_rate=0.075),
                  metrics=[tf.keras.metrics.MeanAbsolutePercentageError(), tf.keras.metrics.RootMeanSquaredError()])

    history = model.fit(trainX, trainY, epochs=MAX_EPOCHS,
                        validation_data=validData,
                        callbacks=[early_stopping])
    return history


multi_step_dense = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=2),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    #     tf.keras.layers.Reshape([1, -1]),
])

history = compile_and_fit(multi_step_dense, X_train, y_train, (X_val, y_val), MAX_EPOCHS=50)

# %%

multi_linear_model = tf.keras.Sequential([
    # Take the last time-step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, out_steps*features]
    tf.keras.layers.Dense(1 * 2,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([1, 2])
])

history = compile_and_fit(multi_linear_model, X_train, y_train, (X_val, y_val), MAX_EPOCHS=50)

# %%

np.array(X_train).shape

# %%

np.array(y_train).shape
