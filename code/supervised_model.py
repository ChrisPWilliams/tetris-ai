import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tetfile as tfile


def BuildAndTrain(fileID):
    #training data load and preprocessing
    file = tfile.tfile(fileID)
    raw_input = file.read()
    raw_array = np.array(raw_input)
    samples = len(raw_array)
    commands = np.copy(raw_array[:,1])
    commands = commands.astype("float32")
    states = np.copy(raw_array[:,0])
    states = np.stack(states)                               #don't want an array of arrays, need to just add a dimension
    states = states.reshape(samples,250).astype("float32")
    x_train = states[0:samples-200]
    y_train = commands[0:samples-200]
    x_val = states[samples-200:samples]
    y_val = commands[samples-200:samples]
    
    #build model
    inputs = keras.Input(shape=(250,), name="screens")
    x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(5, activation="softmax", name="command_predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    #compile with appropriate settings
    model.compile(
        optimizer=keras.optimizers.RMSprop(),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.SparseCategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    print("Fitting model on training data")
    history = model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=2,
        #pass validation for monitoring each epoch
        validation_data=(x_val, y_val),
    )
    return model

