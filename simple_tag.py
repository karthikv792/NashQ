from pettingzoo.mpe import simple_tag_v2
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
from env import GridWorld
from nashq_agent import NashQLearner
from tqdm import tqdm

"""
TODO: 
"""


def run_episode(env, episode, agents, agents_names, learning=True, max_steps=1000):
    """Run single episode of the game"""
    state = env.state
    prev_observations = [None, None]
    prev_rewards = [None, None]
    prev_actions = [None, None]
    step = 0
    for agent in env.agent_iter():
        observation, reward, done, infos = env.last()
        curr_agent = agents_names.index(agent)
        prev_observations[curr_agent] = observation
        prev_rewards[curr_agent] = reward
        action = agents[curr_agent].choose_action(tuple(observation), learning)
        prev_actions[curr_agent] = action
        if done:
            action = None
        env.step(action)
        # print("Curr agent",curr_agent,"\n")
        if curr_agent == 1:
            agents[0].learn(tuple(prev_observations[0]), prev_rewards[0], prev_rewards[1], prev_actions[1], learning)
            agents[1].learn(tuple(prev_observations[1]), prev_rewards[1], prev_rewards[0], prev_actions[0], learning)
        step+=1




    average_rewards = []
    average_rewards.append(np.mean(agents[0].reward_list))
    average_rewards.append(np.mean(agents[1].reward_list))
    return step//2, average_rewards

def visualize(env,episode, iteration, state):
    if (os.path.isdir(str(episode)) == False):
        os.mkdir(str(episode))
    board = np.zeros(env.board.shape[:2])
    for ind, i in enumerate(state):
        index = np.unravel_index(i, board.shape)
        board[index] += ind+1
    # create a figure with size 6x6 inches, and 100 dots-per-inch and save it as a PNG file
    plt.figure(figsize=(6, 6), dpi=100)
    plt.imshow(board, cmap='hot', interpolation='nearest')
    plt.savefig(str(episode)+'/nash_q_learning_'+ str(iteration) + '.png')
    plt.close()

if __name__ == '__main__':
    nb_episodes = 500000
    max_steps = 1000

    env = simple_tag_v2.env(num_good = 1, num_adversaries = 1, num_obstacles = 2, max_cycles=100)
    env.reset()
    agents_names = env.agents

    actions = 5
    init_state = tuple(env.state())
    agent1 = NashQLearner(0,init_state,actions) #Adversary
    agent2 = NashQLearner(1,init_state,actions) #Good
    action_history = []
    reward_history = {0:[],1:[]}
    #Train
    for episode in range(nb_episodes):
        env.reset()
        print("Episode: ",episode,"\n")
        step, rewards = run_episode(env, episode, [agent1, agent2], agents_names, True, max_steps)
        reward_history[0].append(rewards[0])
        reward_history[1].append(rewards[1])
        action_history.append(step)
        if episode % 500 == 0:
            f = open("nash_q_learning_rewards.txt", "w")
            f.write(f"Episode: {episode}\n")
            f.write(f"Action History: {action_history}\n")
            f.write(f"Adversary Reward History: {reward_history[0]}\n")
            f.write(f"Good Reward History: {reward_history[1]}\n")
            f.close()


    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(len(action_history)), action_history, label="step")
    plt.legend()
    plt.subplot(3, 1, 2)
    reward_history["0"] = np.array(reward_history[0])
    reward_history["1"] = np.array(reward_history[1])
    # print(action_history)
    # print(reward_history["0"])
    # print(reward_history["1"])
    plt.plot(np.arange(len(reward_history["0"])),
             reward_history["0"], label="reward_history0")
    plt.plot(np.arange(len(reward_history["1"])),
             reward_history["1"], label="reward_history1")
    plt.xlabel('Episode')
    plt.legend()
    plt.savefig("result_simple_tag.png")
    plt.show()

"""class NashQLearning:
    def __init__(self):
        self.n_states = 10
        self.n_actions = 4
        self.learning_agents = 2
        self.agents = [1, 2]
        self.n_episodes = 1000
        self.n_iterations = 1000
        self.Q = {1: {}, 2: {}}
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 1
        self.iteration = 0
        self.prev_state = None
        self.n = {1: {}, 2: {}}
        self.env = GridWorld()

    def update_epsilon(self):
        self.epsilon *= self.epsilon * 0.999
        if self.epsilon < 0.01:
            self.epsilon = 0.01

    def get_action(self, state, agent):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            # print("Agent: ", agent)
            # print("State: ", state)
            # print("Q: ", self.Q[agent][state])
            # print("Agent, State: ", agent, state)
            a = max(self.Q[agent][state], key=self.Q[agent][state].get)
            # print("Action: ", a)

            return a[agent - 1]

    def check_new_state(self, state, agent):
        if state not in self.Q[agent]:
            self.Q[agent][state] = {}
            for i in range(self.n_actions):
                for j in range(self.n_actions):
                    self.Q[agent][state][(i, j)] = 0


    def update_Q(self, state, action, agent, reward):
        self.Q[agent][self.prev_state][action] = (1 - self.alpha) * self.Q[agent][self.prev_state][action] + self.alpha * (
                        reward + ((self.gamma * self.Q[agent][self.prev_state][action] ) - self.Q[agent][state][action]))

    def visualize(self, episode, iteration, state):
        if (os.path.isdir(str(episode)) == False):
            os.mkdir(str(episode))
        board = np.zeros(self.env.board.shape[:2])
        for ind, i in enumerate(state):
            index = np.unravel_index(i, board.shape)
            board[index] += ind+1
        # create a figure with size 6x6 inches, and 100 dots-per-inch and save it as a PNG file
        plt.figure(figsize=(6, 6), dpi=100)
        plt.imshow(board, cmap='hot', interpolation='nearest')
        plt.savefig(str(episode)+'/nash_q_learning_'+ str(iteration) + '.png')
        plt.close()




    def train(self):
        total_rewards = []
        episodes = []
        for i in tqdm(range(self.n_episodes)):
            self.env.reset()
            self.prev_state = self.env.state
            self.iteration = 0
            total_reward = 0
            # print("------Episode: ", i, "------")
            # print("Number of states: ", self.env.n_states)
            for j in (range(self.n_iterations)):
                self.iteration += 1
                # print("----------------Iteration: ", self.iteration, "----------------")

                action = []
                for agent in self.agents:
                    action.append(self.get_action(self.env.state, agent))
                # print(action)
                state, reward, done, observations = self.env.step(action)
                # print("----------------State: ", state, "----------------")
                print("----------------Reward: ", reward, "----------------")
                # self.visualize(i,self.iteration,state)
                total_reward += sum(reward)
                for agent in self.agents:
                    self.update_Q(state, tuple(action), agent, reward[agent-1])
                self.update_epsilon()
                self.prev_state = state

                # print("-----------------Q-Table-----------------")
                # print(self.Q)
                if done:
                    break
            total_rewards.append(total_reward)
            episodes.append(i)
            seen = set()
            # print(f"\n Episode: {i} Observed States: ", [x for x in self.env.observed_states if x not in seen and not seen.add(x)])
        # Plotting the rewards obtained for each episode
        plt.plot(episodes, total_rewards)
        plt.xlabel('Episodes')
        plt.ylabel('Total Rewards')
        plt.show()

    def test(self):
        self.env.reset()
        self.iteration = 0
        total_reward_agent1 = 0
        total_reward_agent2 = 0
        done = False
        while (not done):
            self.iteration += 1
            action = []
            for agent in self.agents:
                action.append(self.get_action(self.iteration, agent))
            state, reward, done, observations = self.env.step(action)
            self.visualize('test', self.iteration, state)
            total_reward_agent1 += reward[0]
            total_reward_agent2 += reward[1]
        print("Total Reward: ", total_reward_agent1, total_reward_agent2)

    def NashQ(self, state, agent):
        raise NotImplementedError


# Create a basic grid world environment for multi-agent Q-Learning




"""