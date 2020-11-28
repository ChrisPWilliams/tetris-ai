from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("code")
import numpy as np
import tensorflow as tf
import gamelib as gl
import tetris_env as tenv
import box_env as benv

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

sessionID = 27                                      
model_age_steps = 500000
demo_steps = 1000



demo_game_env_py = benv.TetrisGameEnv(sessionID, True)
demo_game_env = tf_py_environment.TFPyEnvironment(demo_game_env_py)

policy_name = "modelpolicy_session_{0}_step_{1}".format(sessionID, model_age_steps)
saved_policy = tf.compat.v2.saved_model.load(policy_name)
policy_state = saved_policy.get_initial_state(batch_size=3)
time_step = demo_game_env.reset()
demo_return = 0
for i in range(demo_steps):
    policy_step = saved_policy.action(time_step, policy_state)
    policy_state = policy_step.state
    time_step = demo_game_env.step(policy_step.action)
    if not time_step.is_last():
        demo_return += time_step.reward
    else:
        print("episode reward = {0}".format(demo_return))
        demo_return = 0
    if demo_game_env_py.game.dqn_quitcheck():
        break