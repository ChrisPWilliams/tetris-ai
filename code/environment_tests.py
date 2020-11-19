import pygame as pg
import numpy as np
import tensorflow as tf
import tetfile as tfile
import gamelib as gl

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
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

sessionID = 0
model = 0                                      
demo = False                                     #DO WE SHOW MACHINE PLAY?

class TetrisGameEnv(py_environment.PyEnvironment):

    def __init__(self):
        self.game = gl.TetrisGame(sessionID,False,demo)
        self._action_spec = array_spec.BoundedArraySpec(shape=(1,), dtype=np.int32, minimum=0, maximum=4, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(25,10), dtype=np.int32, minimum=0, maximum=2, name='observation')
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.game.score = 0
        self.game.stepcount = 0
        self.game.screenmat = np.zeros((25,10))
        self._episode_ended = False
        return ts.restart(self.game.screenmat.astype("int32"))

    def _step(self, action):

        if self._episode_ended:             # The last action ended the episode. Ignore the current action and start a new episode.
            return self.reset()
            # Make sure episodes don't go on forever.
        instructionint = action[0]
        self.game.instruction = gl.decode[instructionint]
        status = self.game.dqn_update()
        if status == "end_episode" or self.game.stepcount == 1000:
            self._episode_ended = True
        reward = self.game.stepcount + 10*self.game.score        #score for clearing lines is weighted 10x compared to just keeping the game going for one more move
        if self._episode_ended:
            return ts.termination(self.game.screenmat.astype("int32"), reward)
        else:
            return ts.transition(self.game.screenmat.astype("int32"), reward, discount=1.0)

# SETUP ENVIRONMENTS

# test_game_env_py = TetrisGameEnv()
# test_game_env = tf_py_environment.TFPyEnvironment(test_game_env_py)                   
