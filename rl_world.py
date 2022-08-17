import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import random
from gym import spaces
from gym.utils import seeding


class Navigation(object):
    def __init__(self, edge_length, num_objects):
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
        Later on the observation will be one-hot encoded into (edge_length**2, 2); each column is a one-hot vector. The first column is current object, the
        second column is goal object.

        Action space: 5 possible action: 0 = standing still, 1 = up, 2 = right, 3 = down, 4 = left

        """
        self.reward = 0
        self.edge_length = edge_length
        self.num_objects = num_objects
        self.num_locations = edge_length ** 2
        self.done = False
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict(
            {'current_object': spaces.MultiBinary(self.num_objects),
             'goal_object': spaces.MultiBinary(self.num_objects)})
        self.grid = np.arange(self.num_locations).reshape((edge_length, edge_length))
        self.init_location = None
        self.goal_location = None
        self.init_object = None
        self.goal_object = None
        self.current_location = None
        self.current_object = None
        self.observation = None
        self.location_to_object = None
        self.shortest_distance = None
        self.node_visit_counter = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def env_reset(self):
        # reset goal location
        self.goal_location = np.random.randint(self.num_locations)
        # randomly assign objects to locations
        self.location_to_object = np.eye(self.num_objects)[np.random.randint(self.num_objects, size=self.num_locations)]
        self.goal_object = self.location_to_object[self.goal_location]
        self.node_visit_counter = np.zeros(self.num_locations)


    def trial_reset(self):
        # Pick a random location
        self.init_location = np.random.randint(self.num_locations)
        self.init_object = self.location_to_object[self.init_location]
        self.current_location = self.init_location
        self.current_object = self.init_object
        self.observation = {'current_object': self.current_object,
                            'goal_object': self.goal_object}  # integers. Need to be translated to one-hot.
        self.shortest_distance = abs(
            int(np.where(self.grid == self.goal_location)[0] - np.where(self.grid == self.init_location)[0])) + abs(
            int(np.where(self.grid == self.goal_location)[1] - np.where(self.grid == self.init_location)[1]))
        self.node_visit_counter[self.current_location] += 1

    def step(self, action):
        """
        :param action:
        :return: observation, reward, done, info
        """
        assert self.action_space.contains(action)
        # step
        if action == 1:  # up
            if self.current_location - self.edge_length < 0:  # Walks into top border
                next_location = self.current_location
            else:
                next_location = self.current_location - self.edge_length
        elif action == 2:  # right
            if (self.current_location + 1) % 5 == 0:  # Walks into right border
                next_location = self.current_location
            else:
                next_location = self.current_location + 1
        elif action == 3:  # down
            if self.current_location + self.edge_length > (self.num_locations - 1):  # walks into bottom border
                next_location = self.current_location
            else:
                next_location = self.current_location + self.edge_length
        elif action == 4:  # left
            if self.current_location % 5 == 0:  # walks into left border
                next_location = self.current_location
            else:
                next_location = self.current_location - 1
        elif action == 0:  # stay still
            next_location = self.current_location

        assert next_location < self.num_locations, "Next location must be between 0 and num_locations"

        self.current_location = next_location
        self.node_visit_counter[self.current_location] += 1

        # if current object = goal object: finish trial
        if self.current_location == self.goal_location:
            self.done = True
            self.reward = 1
        else:
            self.done = False
            self.reward = -0.1

        self.observation = {'current_object': self.location_to_object[self.current_location],
                            'goal_object': self.goal_object}

        return self.observation, self.reward, self.done, {}

    def render(self):
        image = 255 * np.ones((self.edge_length, self.edge_length, 3))
        image[self.grid == self.goal_location] = [0, 255, 0]  # Green indicates goal location
        image[self.grid == self.current_location] = [0, 0, 255]  # Blue indicates current location
        plt.imshow(image)
        plt.show()
