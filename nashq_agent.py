import numpy as np
import random
import nashpy as nash
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

    def get_pi(self, state):
        pi, _ = self.compute_pi(state)
        return pi
    def choose_action(self, state, training=True):
        pi_nash = self.compute_pi(state)

        action = None
        if training:
            if np.random.random() < self.epsilon:
                action = random.randint(0, self.actions-1)
            else:
                # a = max(self.Q[state], key=self.Q[state].get)
                action = random.choice(np.flatnonzero(pi_nash[0] == pi_nash[0].max()))
        else:
            # a = max(self.Q[state], key=self.Q[state].get)
            action = random.choice(np.flatnonzero(pi_nash[0] == pi_nash[0].max()))

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
                    self.n[(state, i, j)] = 0


    def compute_pi(self, state):
        pi = []
        pi_opponent = []
        for i in range(self.actions):
            row_q = []
            row_opponent_q = []
            for j in range(self.actions):
                row_q.append(self.Q[state][(i, j)])
                row_opponent_q.append(self.opponent_Q[state][(i, j)])
            pi.append(row_q)
            pi_opponent.append(row_opponent_q)
        # Compute Nash Equilibrium using nashpy with Lemke Howson algorithm
        nash_game = nash.Game(pi, pi_opponent)
        equilibria = nash_game.lemke_howson_enumeration()
        pi_nash = None
        try:
            pi_nash_list = list(equilibria)
        except:
            pi_nash_list = []
        for eq in pi_nash_list:
            if eq[0].shape == (self.actions,) and eq[1].shape == (self.actions,):
                if any(np.isnan(eq[0]))==False and any(np.isnan(eq[1]))==False:
                    pi_nash = (eq[0], eq[1])
                    break
        if pi_nash is None:
            pi_nash = (np.ones(self.actions)/self.actions, np.ones(self.actions)/self.actions)
        return pi_nash



    def getNashQ(self, state, pi, pi_opponent, opponent=False):
        #return max of Q[state]
        nashq = 0
        for a1 in range(self.actions):
            for a2 in range(self.actions):
                if not opponent:
                    nashq += pi[a1] * pi_opponent[a1] * self.Q[state][(a1, a2)]
                else:
                    nashq += pi[a1] * pi_opponent[a1] * self.opponent_Q[state][(a1, a2)]
        return nashq

    def getOpponentQ(self, state):
        #return max of Q[state]
        return self.opponent_Q[state][max(self.opponent_Q[state], key=self.opponent_Q[state].get)]

    def update_alpha(self, state, action_opponent):
        self.alpha = 1/(self.n[(state, self.prev_action, action_opponent)])
        if self.alpha < 0.01:
            self.alpha = 0.01

    def learn(self, state, reward, reward_opponent, action_opponent,  training=True):
        self.check_new_state(state)
        pi, pi_opponent = self.compute_pi(state)
        if training:
            # print("Learning")
            # print("Q", self.Q)
            self.n[(state, self.prev_action, action_opponent)] += 1
            nashq = self.getNashQ(state, pi, pi_opponent)
            opponentq = self.getNashQ(state, pi_opponent, pi, True)
            action = tuple((self.prev_action, action_opponent))
            action_o = tuple((action_opponent, self.prev_action))
            self.update_alpha(state, action_opponent)
            self.Q[self.prev_state][action] = self.Q[self.prev_state][action] + self.alpha * (reward + (self.gamma * nashq) - self.Q[self.prev_state][action])
            self.opponent_Q[self.prev_state][action_o] = self.opponent_Q[self.prev_state][action_o] + self.alpha * (reward_opponent + (self.gamma * opponentq) - self.opponent_Q[self.prev_state][action_o])
            self.update_epsilon()
        self.reward_list.append(reward)
        self.prev_state = state
