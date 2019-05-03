import numpy as np
import random
import copy
from collections import namedtuple, deque

import os

#from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3#1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4 #1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4 #1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0")# if torch.cuda.is_available() else "cpu")



class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, 
        device,
        state_size, n_agents, action_size, random_seed):  
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
       
        
        self.n_agents = n_agents
        self.state_size = state_size #size of state space
        self.action_size = action_size #size of action space
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device) #redular NN for direct update
        self.actor_target = Actor(state_size, action_size, random_seed).to(device) #target network for Fixed-Q-Targets, will be updated with soft update
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR) #optimizer with learning rate "LR_Actor

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device) #same for Critic NN
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY) #weight_decay: L2 regularization, to reduce over-fitting

        # Noise process
        self.noise = OUNoise((n_agents, action_size), random_seed) #to add to Action space. To further improve, add to NN parameters: https://openai.com/blog/better-exploration-with-parameter-noise/

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, state, action, reward, next_state, done):#move one timestep ahead
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for i in range(self.n_agents):
            self.memory.add(state[i,:], action[i,:], reward[i], next_state[i,:], done[i]) # store current SARS tuple in replay buffer

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:#if enough SARS tuples in replay buffer 
            experiences = self.memory.sample()#take a random sample of the replay buffer (=memory)
            self.learn(experiences, GAMMA) #GAMMA = discount factor; train local (regular) NNs (actor and critic) from experiences

    def act(self, state, add_noise=True): # retrieve action, by predicting with current model / policy
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device) #convert state from numpy to pyTorch as float and move it to device memory (GPU)
        self.actor_local.eval() #set model in eval mode. this is used to predict something
        #eval mode is required when changing the NN, e.g. when using a dropout layer, in which neurons are set to 0
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy() #feed state(observables) into NN to predict output
            #.cpu() moves tensor to cpu, as some operations cannot be performed on gpu
            #.data.numpy() translates torch tensor into numpy array
            
            
        self.actor_local.train()#set model back to train mode, so weights become writeable
        if add_noise:
            action += self.noise.sample() #add noise to the action space to improve performance of learning
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        #DDPG How it works
        #the Actor NN predicts the action to take next. It serves as the "maximizer" over the Q-Values.
        #The Critic NN is only required in training.
        #as input the critic takes the Q-Values (state-action values) of the next state (given by current policy)
        #output is an adjustment to the Q-Values. It thereby serves like a randomness generator. (which is switched off once done = 1)
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states) #target(from Fixed Q networks) actor NN is used to predict the corresponding next actions to the next_states
        Q_targets_next = self.critic_target(next_states, actions_next) #predict Q-Values of current state (that means the state-action values of current state)      
        # (that can serve as randomness for better training)
        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones)) #Q-values(state-action values) of current state = current state rewards + discounted reward of future state
        #when training is finished(done=1) no input from Critic is used
        
        
        # Compute critic loss
        Q_expected = self.critic_local(states, actions) #current Q-values(state-action values) with local(regular Fixed Q) critic-NN
        critic_loss = F.mse_loss(Q_expected, Q_targets) #this loss is a positive number, thats why there is no minus sign in front in order for gradient descent; mse = mean-squared-error, mean squared error over the complete experience batch
        # Minimize the loss
        self.critic_optimizer.zero_grad() #clear partial derivatives in optimizer, from previous steps
        critic_loss.backward() #compute partial derivative for every weight / parameter to MINIMIZE(thats why there is no minus sign) CRITIC_LOSS that can affect the overall gradient, and has requires_grad=True (seems that this is the default setting for weights)
        self.critic_optimizer.step() #update the weights in direction of positive gradient
        #optimizer ist nur auf critic_local bezogen (in init) and does gradient descent by default (daher das minus)

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states) #local(regular in Fixed Q-Targets) NN (which is trained within this batch) of actor predicts actions for current states (from experience batch)
        actor_loss = -self.critic_local(states, actions_pred).mean() 
        #using this as loss will lead to minimization of Q_targets_next. If this(without the minus) is minimized(maximized with minus), the critic is not required anymore, because the 
        #optimal Q-Values (state-action values) ,for the actor, to choose an action, are given by the rewards(that come from the environment)
        
        #!!!!!!!!!!!!!!!!self.critic_local(state..) kann nicht kleiner 0 sein, da Q-values immer grosser 0)--> optimizer macht immer gradient ascent!!!!!!!!!!!!!!!!!
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()#clear partial derivatives in optimizer, from previous steps
        actor_loss.backward()#compute partial derivative for every weight / parameter to MINIMIZE(thats why there is no minus sign) ACTOR_LOSS
        self.actor_optimizer.step()#update the weights in direction of positive gradient

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)#soft update des critic target NN
        self.soft_update(self.actor_local, self.actor_target, TAU)  #soft update des actor target NN                   

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size) #returns tuple of size "size" random values
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)#add to experience e to replay buffer
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)#take Batch size number of random elements(that can be arrays, here SARS tuples) from the replay buffer

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        #numpy vstack is used to stack sequence of input arrays vertically to make a single array
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)