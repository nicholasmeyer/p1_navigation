import time
import os.path
import argparse
from collections import deque

from dqn_agent import Agent

import torch
import numpy as np
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

# initialize environment
env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
# action_size is the dimension of the action space
action_size = brain.vector_action_space_size
# state_size is the dimension of the state space
state_size = brain.vector_observation_space_size

def dqn(n_episodes,
        max_t,
        eps_start,
        eps_end,
        eps_decay,
        seed,
        buffer_size,
        batch_size,
        gamma,
        tau,
        lr,
        update_every):

    """Deep Q-Learning.

        Params
        ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        seed (int): initialize pseudo random number generator
        buffer_size (int): maximum size of buffer
        batch_size (int): size of each training batch
        gamma (float): discount factor
        tau (float): interpolation parameter
        lr (int): learning rate
        update_every (int): learn every UPDATE_EVERY time steps

        Returns
        =======
        None
    """
    if not (0. <= eps_start <= 1.0) and (0. <= eps_end <= 1.0):
        print("epsilon for an epsilon greedy strategy should be in [0,1]")
        sys.exit(1)
    agent = Agent(state_size,
                  action_size,
                  seed,
                  buffer_size,
                  batch_size,
                  gamma,
                  tau,
                  lr,
                  update_every)
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0

        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    # training plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid()
    ax.plot(np.arange(len(scores)), scores)
    ax.set(xlabel="Episode #", ylabel="'Score", title="Double Deep Q Network")
    fig.savefig("DoubleDeepQNetwork.pdf")

def trained_agent():
    """Game play using the trained agent"""
    if not os.path.isfile('checkpoint.pth'):
        print("please train the agent before calling this method")
        sys.exit(1)
    agent = Agent(37, 4,  0, 0, 0, 0, 0, 0, 0)
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    while True:
        action = agent.act(state)
        env_info = env.step(action)[brain_name]
        time.sleep(0.2)
        next_state = env_info.vector_observations[0]
        done = env_info.local_done[0]
        state = next_state
        if done:
            break
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Udacity Deep Reinforcement Learning Nano Degree - Project 1 Navigation')

    parser.add_argument('--n_episodes', metavar='', type=int, default=10000, help='maximum number of training episodes')
    parser.add_argument('--max_t', metavar='', type=int, default=1000,
                help='maximum number of timesteps per episode')
    parser.add_argument('--eps_start', metavar='', type=float, default=1.0, help='starting value of epsilon, for epsilon-greedy action selection')
    parser.add_argument('--eps_end', metavar='', type=float, default=0.01, help='minimum value of epsilon')
    parser.add_argument('--eps_decay', metavar='', type=float, default=0.995, help='multiplicative factor (per episode) for decreasing epsilon')
    parser.add_argument('--seed', metavar='', type=int, default=0, help='seed for stochastic variables')
    parser.add_argument('--buffer_size', metavar='', type=int, default=int(1e5), help='replay buffer size')
    parser.add_argument('--batch_size', metavar='', type=int, default=64, help='minibatch size')
    parser.add_argument('--gamma', metavar='', type=float, default=0.99, help='discount factor')
    parser.add_argument('--tau', metavar='', type=float, default=1e-3, help='for soft update of target parameters')
    parser.add_argument('--lr', metavar='', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--update_every', metavar='', type=int, default=4, help='how often to update the network')
    parser.add_argument('--train_test', metavar='', type=int, default=0, help='0 to train and 1 to test agent')
    args = parser.parse_args()

    if args.train_test == 0:
        dqn(args.n_episodes,
        args.max_t,
        args.eps_start,
        args.eps_end,
        args.eps_decay,
        args.seed,
        args.buffer_size,
        args.batch_size,
        args.gamma,
        args.tau,
        args.lr,
        args.update_every)
    elif args.train_test == 1:
        trained_agent()
    else:
       print("invalid argument for train_test, please use 0 to train and 1 to test agent")
       sys.exit(1)

