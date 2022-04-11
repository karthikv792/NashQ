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

class NashQLearning:
    def __init__(self):
        self.n_states = 10
        self.n_actions = 4
        self.learning_agents = 4
        self.n_episodes = 1000
        self.Q = np.array([np.zeros((self.n_states, self.n_actions, self.learning_agents)) for i in self.learning_agents])

        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        self.iteration = 0

    def get_action(self, state, agent):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(self.Q[agent][state, :, agent])

    def get_state(self, agent):
        #Get from environment
        return random.randint(0, self.n_states - 1)

    def get_reward(self, state, action, agent):
        #Get from environment
        if state == self.n_states - 1:
            return 1
        else:
            return 0

    def update_Q(self, state, action, agent, reward):
        self.Q[agent][state, action, agent] = (1 - self.alpha)*self.Q[agent][state, action, agent] + self.alpha * (reward + self.gamma * self.NashQ(state, agent))

    def NashQ(self, state, agent):


