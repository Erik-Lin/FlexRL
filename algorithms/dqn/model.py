import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim):
		"""
		:param state_dim: Dimension of input state (int)
		:param action_dim: Dimension of output action (int)
		"""
		super(Actor, self).__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim

		self.fc1 = nn.Linear(state_dim,1024,dtype=torch.float)
		self.fc2 = nn.Linear(1024,1024,dtype=torch.float)
		self.fc3 = nn.Linear(1024,512,dtype=torch.float)
		self.fc4 = nn.Linear(512,action_dim,dtype=torch.float)

	def forward(self, state):
		"""
		returns policy function Pi(s) obtained from actor network
		this function is a gaussian prob distribution for all actions
		with mean lying in (-1,1) and sigma lying in (0,1)
		The sampled action can , then later be rescaled
		:param state: Input state (Torch Variable : [n,state_dim] )
		:return: Output action (Torch Variable: [n,action_dim] )
		"""
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		action_prob = F.softmax(self.fc4(x), dim=-1)
		return action_prob

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        :param state_dim: Dimensions of state (int)
        :param action_dim: Dimension of action (int)
        """
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fcs1 = nn.Linear(state_dim , 1024,dtype=torch.float)
        self.fcs2 = nn.Linear(1024 , 512,dtype=torch.float)
        self.fc3 = nn.Linear(512, 1,dtype=torch.float)

    def forward(self, state, action):
        """
        returns Value function Q(s,a) obtained from critic network
        :param state: Input state (Torch Variable : [n,state_dim] )
        :param action: Input Action (Torch Variable : [n,action_dim] )
        :return: Value function : Q(S,a) (Torch Variable : [n,1] )
        """
        s1 = F.relu(self.fcs1(state))
        x = F.relu(self.fcs2(s1))
        x = self.fc3(x)

        return x