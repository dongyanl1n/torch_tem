import gym
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import random
from gym import spaces
from gym.utils.renderer import Renderer
import pygame
from gym.utils import seeding


class TEM_Navigation(object):
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
        # self.seed()

    #def seed(self, seed=None):
    #    self.np_random, seed = seeding.np_random(seed)
    #    return [seed]

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


class SimpleNavigation(gym.Env):  # COPIED FROM https://github.com/Farama-Foundation/gym-examples/blob/main/gym_examples/envs/grid_world.py
    metadata = {"render_modes": ["human", "rgb_array", "single_rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._renderer = Renderer(self.render_mode, self._render_frame)

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, return_info=False, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        self._renderer.reset()
        self._renderer.render_step()

        return (observation, info) if return_info else observation

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        done = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if done else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        self._renderer.render_step()

        return observation, reward, done, info

    def render(self):
        return self._renderer.get_renders()

    def _render_frame(self, mode):
        assert mode is not None

        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
                ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
            )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


def render_observation_as_image(size, agent_loc, target_loc, show_target=True):
    image = np.ones((size, size, 3))*255  # white background
    image[agent_loc[0], agent_loc[1]] = [0, 0, 255]  # blue for agent location
    if show_target:
        image[target_loc[0], target_loc[1]] = [255, 0, 0]  # red for goal location
    return image