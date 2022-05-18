# Implement the Nash Q Learning algorithm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import time
import sys
import os
import math
import copy

from tqdm import tqdm

"""
TODO: 
"""


class GridWorld:
    def __init__(self, goal_pos = [(0, 1),(0, 1)]):
        super().__init__()
        self.agents = [1, 2]
        self.board = np.array([[(0, 0), (0, 0), (0, 0)], [(0, 0), (0, 0), (0, 0)], [(1, 0), (0, 0), (0, 1)]])
        self.state = self.get_state(self.board)
        self.n_states = (self.board.shape[0] * self.board.shape[1]) ** len(self.agents)
        # print("------------Number of states: ", self.n_states, "------------")
        self.n_actions = 4
        self.actions = ["up", "down", "left", "right"]
        self.goal = goal_pos
        self.observations = {1: {'actions': [], 'rewards': []}, 2: {'actions': [], 'rewards': []}}
        self.observed_states = []

    # (2,2) => 8
    def get_agent_index_ravel(self, agent_index, board):
        return np.ravel_multi_index(agent_index, board.shape[:2])
    # 8 => (2,2)
    def get_agent_index_unravel(self, agent_index, board):
        return np.unravel_index(agent_index, board.shape[:2])

    def get_state(self, board):
        # Get the state of the environment
        state = []
        for index, agent in enumerate(self.agents):
            for x, i1 in enumerate(board):
                for y, j in enumerate(i1):
                    if j[index] == 1:
                        state.append(np.ravel_multi_index((x, y), board.shape[:2]))
        return tuple(state)

    def reset(self, goal_pos = [(0, 1),(0, 1)]):
        self.agents = [1, 2]
        self.board = np.array([[(0, 0), (0, 0), (0, 0)], [(0, 0), (0, 0), (0, 0)], [(1, 0), (0, 0), (0, 1)]])
        self.state = self.get_state(self.board)
        self.n_states = (self.board.shape[0] * self.board.shape[1]) ** len(self.agents)
        # print("------------Number of states: ", self.n_states, "------------")
        self.n_actions = 4
        self.actions = ["up", "down", "left", "right"]
        self.goal = goal_pos
        self.observations = {1: {'actions': [], 'rewards': []}, 2: {'actions': [], 'rewards': []}}
        self.observed_states = []

    def get_new_pos(self, action, agent):
        # Search for agent in the board
        # Get the position of the agent in board
        agent_pos = self.get_agent_index_unravel(self.state[agent - 1], self.board)
        opponent_pos = self.get_agent_index_unravel(self.state[agent%2], self.board)

        # print("------------Agent position: ", agent_pos, "------------")
        # print("------------Action: ", self.actions[action], "------------")
        # Update the board with action
        if action == 0:
            if agent_pos[0] - 1 >= 0:
                return tuple((agent_pos[0] - 1, agent_pos[1]))
        elif action == 1:
            if agent_pos[0] + 1 < self.board.shape[0]:
                return tuple((agent_pos[0] + 1, agent_pos[1]))
        elif action == 2:
            if agent_pos[1] - 1 >= 0:
                return tuple((agent_pos[0], agent_pos[1] - 1))
        elif action == 3:
            if agent_pos[1] + 1 < self.board.shape[1]:
                return tuple((agent_pos[0], agent_pos[1] + 1))

        return agent_pos

    def change_board(self, agent_pos):
        # print("------------Agent position: ", agent_pos, "------------")
        temp_board = np.zeros(self.board.shape)
        for index, i in enumerate(agent_pos):
            temp_board[i][index] = 1
        self.board = temp_board

    def get_reward(self, agent_pos):
        # print("------------Agent position: ", agent_pos, "------------")
        # print("Goal: ", self.goal==agent_pos)
        reward = []
        for index,i in enumerate(agent_pos):
            if i == self.goal[index]:
                reward.append(100)
            else:
                if agent_pos[0]==agent_pos[1] and i!=self.goal[(index+1)%2]:
                    reward.append(-1)
                else:
                    reward.append(0)
        return reward

    # Action is a vector of length 2 with first element corresponding to agent 1 and second element corresponding to agent 2
    def step(self, action):
        # Based on the action, update the state of the environment
        new_agent_pos = []  # Unraveled index
        prev_agent_pos = [self.get_agent_index_unravel(i, self.board) for i in self.state]
        for index, i in enumerate(self.agents):
            new_agent_pos.append(self.get_new_pos(action[index], i))
            self.observations[self.agents[index]]['actions'].append(action[index])
        # If both the agents are in the same position, then the state is the same
        reward = self.get_reward(new_agent_pos)
        # print("1",new_agent_pos, new_agent_pos[0] == new_agent_pos[1])
        if sum(reward) == -2:
            new_agent_pos[0] = self.get_agent_index_unravel(self.state[0], self.board)
            new_agent_pos[1] = self.get_agent_index_unravel(self.state[1], self.board)
        # print("2",new_agent_pos, new_agent_pos[0] == new_agent_pos[1])
        for index, i in enumerate(self.agents):
            self.observations[i]['rewards'].append(reward[index])
        self.change_board(new_agent_pos)
        self.state = self.get_state(self.board)
        self.observed_states.append(self.state)
        if sum(reward) == 200:
            return self.state, reward, True, self.observations
        else:
            return self.state, reward, False, self.observations
