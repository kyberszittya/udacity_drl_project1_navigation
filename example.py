from unityagents import UnityEnvironment
import numpy as np
from dqn_agent import DqnAgent
import torch
from collections import deque
import time
import matplotlib.pyplot as plt

import sys
import os





def run(env, brain_name, agent, episodes=2000, max_t=1000, eps_max=1.0, eps_min=0.04, eps_decay=0.995, chkp="chkp"):

    scores = []
    epsilons = []
    window_size = 100
    scores_window = deque(maxlen=window_size)
    eps = eps_max
    prev_scores = np.nan
    for e in range(1, episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        s = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            a = agent.act(s, eps)
            env_info = env.step(int(a))[brain_name]
            sn = env_info.vector_observations[0]
            r = env_info.rewards[0]
            done = env_info.local_done[0]

            agent.step(s, a, r, sn, done)
            s = sn
            score += r
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        if prev_scores == np.nan:
            prev_scores = score
        eps = max(eps_min, eps * eps_decay)
        epsilons.append(eps)
        if e % window_size == 0:
            print("{}# {} step AVG: {:.2f}".format(e, window_size, np.mean(scores_window)))
        if np.mean(scores_window) >= 13.0:
            print("{}# step AVG: {:.2f}".format(e, np.mean(scores_window)))
            torch.save(agent.q_local.state_dict(), f'{chkp}.pth')
            break
    return scores, epsilons


def main():

    #
    env = UnityEnvironment(file_name="./Banana_Windows_x86_64/Banana.exe")
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    cnt_run = 1

    runs = []
    # DQN agent (ReLU)
    total_time = 0.0
    min_time = float('inf')
    max_time = 0
    for i in range(cnt_run):
        print("Run ReLU: {}".format(i))
        agent = DqnAgent(state_size=37, action_size=4, seed=9090, mode=0)
        t_start = time.time()
        scores, epsilons = run(env, brain_name, agent, chkp="chkp_relu")
        t_end = (time.time() - t_start) / 60
        print("TIME: {:.1f}".format(t_end))
        runs.append([scores, t_end])
        total_time += t_end
        min_time = min(min_time, t_end)
        max_time = max(max_time, t_end)
        with open("./data/{}_ReLU.txt".format(i), "w") as f:
            f.write("{}\n".format(t_end))
            for s in scores:
                f.write("{}\n".format(s))
    # Plot results
    fig = plt.figure(figsize=(8,8))

    for r in runs:
        plt.plot(np.arange(len(r[0])), r[0], label="{}".format(r[1]))
    print("Average time: {}".format(total_time/cnt_run))
    print("Min time: {}".format(min_time))
    print("Max time: {}".format(max_time))
    print("Total time: {}".format(total_time))
    plt.grid(True)
    plt.legend()
    plt.show()

    runs = []
    # DQN agent ELu
    total_time = 0.0
    min_time = 198
    max_time = 0
    for i in range(cnt_run):
        print("Run ELU: {}".format(i))
        agent = DqnAgent(state_size=37, action_size=4, seed=9090, mode=1)
        t_start = time.time()
        scores, epsilons = run(env, brain_name, agent, chkp="chkp_elu")
        t_end = (time.time() - t_start) / 60
        print("TIME: {:.1f}".format(t_end))
        runs.append([scores, t_end])
        total_time += t_end
        min_time = min(min_time, t_end)
        max_time = max(max_time, t_end)
        with open("./data/{}_ELU.txt".format(i), "w") as f:
            f.write("{}\n".format(t_end))
            for s in scores:
                f.write("{}\n".format(s))
    # Plot results
    fig = plt.figure(figsize=(8, 8))

    for r in runs:
        plt.plot(np.arange(len(r[0])), r[0], label="{}".format(r[1]))
    print("Average time: {}".format(total_time / cnt_run))
    print("Min time: {}".format(min_time))
    print("Max time: {}".format(max_time))
    print("Total time: {}".format(total_time))
    plt.grid(True)
    plt.legend()
    plt.show()

    env.close()


if __name__=="__main__":
    main()