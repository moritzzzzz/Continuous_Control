# Continuous_Control
DRL-agent using the Deep Deterministic Policy Gradient method

# DeepRL_Banana
DRL Agent to collect Bananas in Unity environment


# Project: Navigation in a Banana World

### Intro

[//]: # (Image References)

[image1]: https://s3.amazonaws.com/video.udacity-data.com/topher/2018/June/5b1ea778_reacher/reacher.gif "Trained Agent"


This projects goal is to utilize Deep Reinforcement Learning (DRL) to train an agent to navigate through 2 dimensional environment and collect yellow bananas, while avoiding to collect blue bananas.

A trained agent will can be seen in below image: 

![Trained Agent][image1]

(source: https://github.com/udacity/deep-reinforcement-learning/blob/master/p1_navigation/README.md)

### Environment
In RL the environment defines what the agent will learn. In this case the environment allows the agent to choose between 4 actions in each timesequence:

- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The state space in contrast is not discrete, but continuous and is perceived by the agent in 37 dimensions. (37 continuous input features)

The rewards of the environment, which serve as reinforcement for the agent to learn, are assigned when its rules are fullfilled:

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thereby the agent will learn to avoid blue bananas while collecting yellow bananas.

The task is episodic. To solve the problem the average score must exceed +13 for at least 100 episodes.

### How to use this Github repository to train an agent to solve the banana world

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
- Get Unity Environment designed for this Banana project.
   -  Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip).
    - Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Navigation.ipynb` to train the agent!

### Expected Result
After approximately 2600 training episodes the agent will reach the average score of +13, which defines this environment as solved.
The current model will allow the agent to improve up to an average score of approximately +16, which can be reached after approximately 4500 episodes.


### Learning from Pixels

The same goal can also be reached by analyzing an image of what is in front of the agent. This image has a resolution of 84x84 in 3 color channels. So overall dimensions 84x84x3.

There are 2 approaches to this:

#### Easy approach:

Take the same agent and same model (fully connected layers). For this to not overload your memory, the state space needs to be shrinked. 
84x84x3 = 21168 features. This combined with multiple fully connected layers will consume vast amounts of memory.
Therefore the central region of the image (state) is cropped and 2 color channels are removed. The result is then flattened to a 1-d vector that can serve as input for the model (of fully connected layers).

#### New approach:

Change the model to use convolutional layers, instead of fully connected layers. Convolutional layers make sense, as we are trying to extract information from image data. Good thing: We only would have to normalize the pixel values, but the dimensions of the image array (state) can be fed without adjustment into the model. Therfore merely the learning function in the Jupyter notebook and the model in model.py would have to be altered.

This should yield a better performance than the easy approach.

Problem: I did not yet implement this second option.

### Instructions to reproduce approach 1

Download a new Unity environment.  This environment is almost identical to the project environment, where the only difference is that the state is an 84 x 84 RGB image, corresponding to the agent's first-person view. 

Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

Place the file in the `p1_navigation/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Navigation_Pixels.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

Follow the instructions in `Navigation_Pixels.ipynb`
