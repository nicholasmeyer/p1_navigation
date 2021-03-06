[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the p1_navigation GitHub repository and unzip (or decompress) the file. 

### Instructions

```
Usage: 

python navigation.py [--n_episodes] <n_episodes> 
                     [--max_t] <max_t> 
                     [--eps_start] <eps_start> 
                     [--eps_end] <eps_end> 
                     [--eps_decay] <eps_decay> 
                     [--seed] <seed> 
                     [--buffer_size] <buffer_size> 
                     [--batch_size] <batch_size> 
                     [--gamma] <gamma> 
                     [--tau] <tau> 
                     [--lr] <lr> 
                     [--update_every] <update_every> 
                     [--train_test] <train_test>

Options: 

    --n_episodes <n_episodes>:     maximum number of training episodes (default=10000)
    --max_t <max_t>:               maximum number of timesteps per episode (default=1000)
    --eps_start <eps_start>:       starting value of epsilon, for epsilon-greedy action selection (default=1.0)
    --eps_end <eps_end>:           minimum value of epsilon (default=0.01)
    --eps_decay <eps_decay>:       multiplicative factor (per episode) for decreasing epsilon (default=0.095)
    --seed <seed>:                 initialize pseudo random number generator (default=0)
    --buffer_size <buffer_size>:   maximum size of buffer (default=1e5)
    --batch_size <batch_size>:     size of each training batch (default=64)
    --gamma <gamma>:               discount factor (default=0.99)
    --tau <tau>:                   interpolation parameter (default=1e-3)
    --lr <lr>:                     learning rate (default=5e-4)
    --update_every <update_every>: learn every UPDATE_EVERY time steps (default=4)
    --train_test <train_test>:     0 to train and 1 to test agent (default=0)
    -h, --help                     show this help message and exit.
``` 
