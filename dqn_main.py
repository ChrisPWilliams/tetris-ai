from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("code")
import numpy as np
import tensorflow as tf
import tetfile as tfile
import tetris_env as tenv

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy, policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

# SET HYPERPARAMETERS

num_iterations = 40000                 # roughly 4000 iterations per minute

initial_collect_steps = 1000
collect_steps_per_iteration = 1
replay_buffer_max_length = 100000

batch_size = 64
learning_rate = 3e-4                    # smaller = slower learning
adam_epsilon = 1e-5                     # bigger = slower learning but higher accuracy
target_update_period = 1               # number of episodes before target network updates
start_epsilon = 0.5
end_epsilon = 0.05


num_eval_episodes = 10
eval_interval = 1000
log_interval = 200
save_interval = 40000
# INITIALISE GAME

sessionID = 20                                      

# SETUP ENVIRONMENTS

train_game_env_py = tenv.TetrisGameEnv(sessionID, False)
eval_game_env_py = tenv.TetrisGameEnv(sessionID, False)
train_game_env = tf_py_environment.TFPyEnvironment(train_game_env_py)
eval_game_env = tf_py_environment.TFPyEnvironment(eval_game_env_py)                      

# INITIALISE AGENT

conv_layer_params = ((16, (4,4), 2), (32, (2,2), 1))     # 2 convolutional layers: one with 16 4x4 filters w/ stride 4, one with 32 2x2 filters w/ stride 1
fc_layer_params = (128, 128)         # 2 hidden layers of 128 neurons each
q_net = q_network.QNetwork(train_game_env.observation_spec(),
                           train_game_env.action_spec(), 
                           conv_layer_params=conv_layer_params, 
                           fc_layer_params=fc_layer_params)

# convolutional layer causes breakage: check tetris_env for more

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, epsilon=adam_epsilon)

train_step_counter = tf.Variable(0)

epsilon = tf.compat.v1.train.polynomial_decay(
    start_epsilon,
    train_step_counter,
    num_iterations,
    end_learning_rate=end_epsilon)

agent = dqn_agent.DdqnAgent(
    train_game_env.time_step_spec(),
    train_game_env.action_spec(),
    q_network=q_net,
    epsilon_greedy=epsilon,
    optimizer=optimizer,
    target_update_period=target_update_period,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()

# DEFINE METRICS ETC. (see tf-agents DQN tutorial)

def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for i in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_ret = total_return / num_episodes
    return avg_ret.numpy()[0]

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_game_env.batch_size,
    max_length=replay_buffer_max_length)

def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer, with copies per 0.005 reward (jury-rigged experience prioritisation weighting)
    for i in range(int(traj.reward // 0.005)):
        buffer.add_batch(traj)

def collect_data(environment, policy, buffer, steps):
    for i in range(steps):
        collect_step(environment, policy, buffer)

# DEFINE POLICIES

eval_policy = agent.policy
collect_policy = agent.collect_policy
random_policy = random_tf_policy.RandomTFPolicy(train_game_env.time_step_spec(), train_game_env.action_spec())


# implement scripted pypolicy to get states from manual trainer and expose as tensorflow graph (see tf-agents policy tutorial)

eval_saver = policy_saver.PolicySaver(agent.policy, batch_size=None)

# initial replay buffer fill
collect_data(train_game_env, random_policy, replay_buffer, initial_collect_steps)

# dataset required to optimise calls to replay buffer
# Dataset generates trajectories with shape [Batchx8x...]
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, 
    sample_batch_size=batch_size, 
    num_steps=2).prefetch(3)
buffer_iterator = iter(dataset)

# TRAIN AGENT

# Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_game_env, agent.policy, num_eval_episodes)
returns = [avg_return]

for i in range(num_iterations):

    # Collect a few steps using collect_policy and save to the replay buffer.
    collect_data(train_game_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(buffer_iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_game_env, agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)

    if step % save_interval == 0:
        savename = "modelpolicy_session_{0}_step_{1}".format(sessionID, step)
        eval_saver.save(savename)

rand_avg_return = compute_avg_return(eval_game_env, random_policy, num_eval_episodes)
print('random benchmark return = {0}'.format(rand_avg_return))