import numpy as np
from numpy import array
import random
from gym import spaces
from gym.utils import seeding


class Navigation(object):
    def __init__(self, edge_length):
        """
        A edge_length by edge_length grid environment in which agent must navigate to the goal location using as short
        path as possible to receive maximal reward and finish trial.

        In each trial:
        - agent will be randomly assigned to a location that is associated with an object (one-hot vector). The object
        but not the starting location is given as state input.
        - agent will also be given which object the reward location is associated with.
        - in each time step, the agent choose from action space 0~4 (stay still, up, right, down, left) # TODO: how does TEM react to borders?
        and move to a new location - again, location is unknown but object is given as state input.
        - each movement is assigned a reward of -1 to ensure shortest path. When current object == goal object,
        reward = 100 and trial ends.

        Observation space: Dict of 'current' (for current observed object) and 'goal' (for goal object), each is integer from 0~24
        Later on the observation will be one-hot encoded into (edge_length^2, 2); each column is a one-hot vector. The first column is current object, the
        second column is goal object.

        Action space: 5 possible action: 0 = standing still, 1 = up, 2 = right, 3 = down, 4 = left

        """
        self.reward = 0
        self.edge_length = edge_length
        self.done = False
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict({'current': spaces.Discrete(edge_length^2), 'goal': spaces.Discrete(edge_length^2)})
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # assign agent to random object


    def step(self, action):
        """
        :param action:
        :return: observation, reward, done, info
        """
        # if current object = goal object: finish trial

        # Take step

        #