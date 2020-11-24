import sys
sys.path.append("code")
import numpy as np
import tensorflow as tf
import gamelib as gl

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts

def one_hot(array):
    encoded = (np.arange(2) == array[...,None]-1).astype("float32")
    return encoded

class TetrisGameEnv(py_environment.PyEnvironment):

    def __init__(self, sessionID, demo):
        self.game = gl.TetrisGame(sessionID,False,demo)
        self.obs = one_hot(self.game.screenmat)
        self._action_spec = array_spec.BoundedArraySpec(shape=(1,), dtype=np.int32, minimum=0, maximum=4, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(25,10,2), dtype=np.float32, minimum=0, maximum=2, name='observation')
        self._episode_ended = False

    def action_spec(self):                              
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.game.game_reset()
        self._episode_ended = False
        return ts.restart(self.obs)

    def _step(self, action):

        if self._episode_ended:             # The last action ended the episode. Ignore the current action and start a new episode.
            return self.reset()
            # Make sure episodes don't go on forever.
        oldscore = self.game.score
        instructionint = action[0]
        self.game.instruction = gl.decode[instructionint]
        status = self.game.dqn_update()
        if status == "end_episode" or self.game.stepcount == 1000:
            self._episode_ended = True
        self.obs = one_hot(self.game.screenmat)
        height = self.game.get_max_height()
        flush_metric = self.game.flush_metric()
        scorediff = self.game.score-oldscore
        if scorediff < 0:
            scorediff = 0
        
        reward = scorediff + 0.01*(1-(height/48)) + 0.002*flush_metric      
        # one point per line cleared, small reward for staying in the game, scaled down by the height of the current tetromino stack, with reward for flush shape placement
        
        if status == "end_episode":                      # extra incentive to avoid hitting the top of the stack as opposed to timing out the session
            reward += -1
        if self._episode_ended:
            return ts.termination(self.obs, reward)
        else:
            return ts.transition(self.obs, reward, discount=0.99)
