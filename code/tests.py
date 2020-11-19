import numpy as np
import tensorflow as tf
import supervised_model as mdl
import tetfile as tfile

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))





# fileID = 0
# model = mdl.BuildAndTrain(fileID)

# file = tfile.tfile(fileID)
# raw_input = file.read()
# raw_array = np.array(raw_input)
# samples = len(raw_array)
# commands = np.copy(raw_array[:,1])
# commands = commands.astype("float32")
# states = np.copy(raw_array[:,0])
# states = np.stack(states)                               #don't want an array of arrays, need to just add a dimension
# states = states.reshape(samples,260).astype("float32")
# x_test = states[samples-100:samples]
# y_test = commands[samples-100:samples]


# Evaluate the model on the test data using `evaluate`
# print("Evaluate on test data")
# results = model.evaluate(x_test, y_test, batch_size=128)
# print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
# print("Generate predictions for 3 samples")
# predictions = model.predict(x_test[:3])
# print("predictions shape:", predictions.shape)
# for i in range(0,3):
#     print("predicted command: {0} with probability {1}".format(np.argmax(predictions[i]), np.amax(predictions[i])))
#     print("actual command: {0}".format(y_test[i]))