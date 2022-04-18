import numpy as np
import random
class NashQLearner:
    def __init__(self, id, init_state, actions, epsilon=1, alpha=0.2, gamma=0.9):
        self.id = id
        self.Q = {}
        self.opponent_Q = {}
        self.actions = actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.prev_state = init_state
        self.prev_action = 0
        self.reward_list = []
        self.n = {}
        self.check_new_state(init_state)

    def reset(self, init_state):
        self.prev_state = init_state
        self.prev_action = 0
        self.reward_list = []
        self.check_new_state(init_state)

    def update_epsilon(self):
        self.epsilon *= self.epsilon * 0.999
        if self.epsilon < 0.01:
            self.epsilon = 0.01


    def choose_action(self, state, training=True):
        action = None
        if training:
            if np.random.random() < self.epsilon:
                action = random.randint(0, self.actions-1)
            else:
                a = max(self.Q[state], key=self.Q[state].get)
                action =  a[0]
        else:
            a = max(self.Q[state], key=self.Q[state].get)
            action =  a[0]

        self.prev_action = action
        return action

    def check_new_state(self, state):
        if state not in self.Q:
            self.Q[state] = {}
            self.opponent_Q[state] = {}
            for i in range(self.actions):
                for j in range(self.actions):
                    self.Q[state][(i, j)] = 0
                    self.opponent_Q[state][(i, j)] = 0
                    self.n[(state, i, j)] = 1


    def getNashQ(self, state):
        #return max of Q[state]
        return self.Q[state][max(self.Q[state], key=self.Q[state].get)]

    def getOpponentQ(self, state):
        #return max of Q[state]
        return self.opponent_Q[state][max(self.opponent_Q[state], key=self.opponent_Q[state].get)]

    def learn(self, state, reward, reward_opponent, action_opponent,  training=True):
        self.check_new_state(state)
        if training:
            # print("Learning")
            # print("Q", self.Q)
            nashq = self.getNashQ(state)
            opponentq = self.getOpponentQ(state)
            action = tuple((self.prev_action, action_opponent))
            action_o = tuple((action_opponent, self.prev_action))
            self.Q[self.prev_state][action] = self.Q[self.prev_state][action] + self.alpha * (reward + (self.gamma * nashq) - self.Q[self.prev_state][action])
            self.opponent_Q[self.prev_state][action_o] = self.opponent_Q[self.prev_state][action_o] + self.alpha * (reward_opponent + (self.gamma * opponentq) - self.opponent_Q[self.prev_state][action_o])
            self.update_epsilon()
        self.reward_list.append(reward)
        self.prev_state = state
