import matplotlib.pyplot as plt
import numpy as np
import json
reward_history = {}
f = open("nash_q_learning_rewards.txt", "r")
for line in f:
    if "Episode" in line:
        episode = int(line.split(":")[1])
    elif "Action History" in line:
        a3 = json.loads(line.split(":")[1])
    elif "Adversary Reward History" in line:
        a1 = json.loads(line.split(":")[1])
    elif "Good Reward History" in line:
        a2 = json.loads(line.split(":")[1])
# a1 = [0,16.66668,16.66668,16.66668,16.66668,16.66668,16.66668,16.66668,16.66668,16.66668,]
# a2 = [98.6,50.0,50.0,50.0,50.0,50.0,50.0,50.0,50.0,50.0]
# a3 = [999,5,5,5,5,5,5,5,5,5]
# a3 = [999, 275, 130, 109, 55, 195, 53, 172, 193, 70, 26]
# a1 = [-0.039,12.94565217, 12.89312977, 19.89090909, 14.23214286,  4.44897959,25.87037037,  9.79190751, 12.81443299, 19.66197183, 11.07407407]
# a2 = [13.661,6.42391304, 10.60305344, 11.70909091,  1.73214286, 5.46938776,  9.2037037,   8.63583815,  7.1443299,  12.61971831, 18.48148148]
reward_history["0"] = a1
reward_history["1"] = a2
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(np.arange(len(a3)), a3, label="No of steps in an episode")
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(np.arange(len(reward_history["0"])),
         reward_history["0"], label="Agent 1 (Predator) rewards")
plt.plot(np.arange(len(reward_history["1"])),
         reward_history["1"], label="Agent 2 (Prey) rewards")
#x axis title
plt.xlabel('Episode Number')
#x axis ticks each tick label represent 500 episodes
plt.legend()
plt.savefig("result_ag.png")
plt.show()