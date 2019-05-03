# Continuous_Control
DRL-agent using the Deep Deterministic Policy Gradient (DDPG) method


# Project: Continuous Control

### Intro

[//]: # (Image References)

[image1]: https://s3.amazonaws.com/video.udacity-data.com/topher/2018/June/5b1ea778_reacher/reacher.gif "Trained Agent"


This projects goal is to utilize Deep Reinforcement Learning (DRL) to train an agent to control 4 joints of a "robotic reach-arm" in order to reach a defined position with its hand.

A trained agent will can be seen in below animation, in which the defined position is marked with a green sphere: 

![Trained Agent][image1]

(source: https://s3.amazonaws.com/video.udacity-data.com/topher/2018/June/5b1ea778_reacher/reacher.gif)

### Environment
In RL the environment defines what the agent will learn. In this case the environment allows the agent to choose the magnitude of 4 dimensionsof its action in each timesequence. The action space is continuous, which poses the essence, as well as the challenge in this project. Every action dimension must be in range -1 to 1.

Also the state space is not discrete, but continuous and is perceived by the agent in 33 dimensions. (33 continuous input features)

The rewards of the environment, which serve as reinforcement for the agent to learn, are assigned when its rules are fullfilled:

A reward of +0.1 is provided each timestep the hand of the reacher is placed in the target sphere. 0 reward if it is not placed inside the target sphere. 

The task is episodic. To solve the problem the average score must exceed +30 for at least 100 consecutive episodes.

In order to reduce the probability that we end up with an agent that chooses to take actions, that do not lead to fast learning, we 20 run agents simultaneously and average the scores.

### How to use this Github repository to train an agent to control the reacher

The following system prerequisites are required to get it running with my instructions:

- Windows 10 64-Bit.
- Anaconda for Windows.
- GPU with CUDA support(this will not run on CPU only, as its explicitly disabled).
    Cudatoolkit version 9.0.

#### Setting up the Anaconda environment

- Set up conda environment with Python >=3.6
	- Conda create –name <env_name> python=3.6
- Install Jupyter Kernel for this new environment.
    - python -m ipykernel install --user --name <Kernel_name> --display-name "<kernel_name>".
- Download ML-Agents Toolkit beta 0.4.0a.
   - https://github.com/Unity-Technologies/ml-agents/releases/tag/0.4.0a
- install it by running following command, with activated conda environment in the directory of ml-agents, that contains the setup.py.
   - pip install -e . .
- install PyTorch.
    - conda install pytorch torchvision cudatoolkit=9.0 -c pytorch.
- Get Unity Environment designed for this project.
   -  Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip).
    - Place the file in the DRLND GitHub repository, in the `p2_continuous_control/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in the Jupyter notebook file `Continuous_control_solution.ipynb` to train the agent!

### Expected Result
After approximately 100 training episodes the agent will reach the average score of +30 for 100 consecutive episodes, which defines this environment as solved.

### Techniques utilized
#### DDPG
In this project an "actor-critic" DDPG, as defined in [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971) by Timothy P. Lillicrap et al. was implemented. 

#### Improved Exploration
To improve exploration of this agent stochastic Ornstein-Uhlenbeck was added to the selected action, which lead to an improved learning curve. 

#### Stable Deep Reinforcement Learning(DRL)
In order to make the DRL agent more stable in regards to auto-correlation of weights adjustments, the "Fixed Q-Targets" method was utilized combined with a "Replay Buffer". We update the target neural networks (NN) wiht a "soft-update" method after each training step. Thereby the target network (see ["Fixed Q-targets"](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12389/11847) ) is iteratively updated with the weights of the trained "regular" NN.
