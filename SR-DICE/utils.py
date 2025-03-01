import numpy as np
import torch
from ppo.algo import core

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.start_ptr = 0
		self.start_size = 0

		self.state = np.zeros(core.combined_shape(max_size, state_dim))
		self.action = np.zeros(core.combined_shape(max_size, action_dim))
		self.next_state = np.zeros(core.combined_shape(max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.start_state = np.zeros((max_size, state_dim))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def add_start(self, state):
		self.start_state[self.start_ptr] = state

		self.start_ptr = (self.start_ptr + 1) % self.max_size
		self.start_size = min(self.start_size + 1, self.max_size)


	def all_start(self, batch_size=2048):
		if batch_size == -1:
			return torch.FloatTensor(self.start_state[:self.start_size]).to(self.device)
		else:
			ind = np.random.randint(self.start_size, size=batch_size)
			return torch.FloatTensor(self.start_state[ind]).to(self.device)


	def sample(self, batch_size):
		ind = np.random.randint(self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)

	def sample_all(self):
		ind = np.arange(self.size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)