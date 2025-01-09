import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Deep_TD import Critic, CriticDiscrete

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder_Decoder(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Encoder_Decoder, self).__init__()

		self.e1 = nn.Linear(state_dim + action_dim, 256)
		self.e2 = nn.Linear(256, 256)

		self.r1 = nn.Linear(256, 1, bias=False)

		self.a1 = nn.Linear(256, 256)
		self.a2 = nn.Linear(256, action_dim)

		self.d1 = nn.Linear(256, 256)
		self.d2 = nn.Linear(256, state_dim)


	def forward(self, state, action):
		l = F.relu(self.e1(torch.cat([state, action], 1)))
		l = F.relu(self.e2(l))
		
		r = self.r1(l)

		d = F.relu(self.d1(l))
		ns = self.d2(d)

		d = F.relu(self.a1(l))
		a = self.a2(d)

		return ns, r, a, l

	def latent(self, state, action):
		l = F.relu(self.e1(torch.cat([state, action], 1)))
		l = F.relu(self.e2(l))
		return l


class SR_DICE(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		mujoco=True,
	):

		self.encoder_decoder = Encoder_Decoder(state_dim, action_dim).to(device)
		self.ed_optimizer = torch.optim.Adam(self.encoder_decoder.parameters(), lr=3e-4)

		if mujoco:
			self.critic = Critic(state_dim, action_dim).to(device)
		else:
			self.critic = CriticDiscrete(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.W = torch.ones(1, 256, requires_grad=True, device=device)
		self.W_optimizer = torch.optim.Adam([self.W], lr=3e-4)

		self.discount = discount
		self.tau = tau

		self.total_it = 0

		self.made_start = False
		self.max_action = max_action
		self.action_dim = action_dim
		self.mujoco = mujoco


	def train_encoder_decoder(self, replay_buffer, batch_size=256):
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		if not self.mujoco:
			action = torch.nn.functional.one_hot(action.to(torch.int64), num_classes=self.action_dim).to(torch.float32)
		recons_next, recons_reward, recons_action, lat = self.encoder_decoder(state, action)
		ed_loss = F.mse_loss(recons_next, next_state) + \
				  0.1 * F.mse_loss(recons_reward, reward) + F.mse_loss(recons_action, action)

		self.ed_optimizer.zero_grad()
		ed_loss.backward()
		self.ed_optimizer.step()


	def train_SR(self, replay_buffer, policy, batch_size=256):
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			next_action,_,_ = policy.step(next_state)
			next_action = torch.FloatTensor(next_action).to(device)
			if self.mujoco:
				next_action = (next_action + torch.randn_like(next_action) *
							   self.max_action * 0.1).clamp(-self.max_action, self.max_action)
				latent = self.encoder_decoder.latent(state, action)
			else:
				one_hot_action = torch.nn.functional.one_hot(action.to(torch.int64), num_classes=self.action_dim).to(torch.float32)
				latent = self.encoder_decoder.latent(state, one_hot_action)

			target_Q = latent + self.discount * not_done * self.critic_target(next_state, next_action)

		current_Q = self.critic(state, action)
		critic_loss = F.mse_loss(current_Q, target_Q)

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def train_OPE(self, replay_buffer, policy, batch_size=512):
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		start_state = replay_buffer.all_start()
		with torch.no_grad():
			start_action,_,_ = policy.step(start_state)
			start_action = torch.FloatTensor(start_action).to(device)
			if self.mujoco:
				start_action = (start_action + torch.randn_like(start_action) *
								self.max_action * 0.1).clamp(-self.max_action, self.max_action)

			Q = self.critic(start_state, start_action)
			self.start_Q = (1. - self.discount) * Q.mean(0)

		start_Q = (self.start_Q * self.W).mean()
		if not self.mujoco:
			action = torch.nn.functional.one_hot(action.to(torch.int64), num_classes=self.action_dim).to(torch.float32)
		b_sQ = (self.encoder_decoder.latent(state, action) * self.W).mean(1).pow(2).mean()
		W_loss = (0.5 * b_sQ - start_Q)

		self.W_optimizer.zero_grad()
		W_loss.backward()
		self.W_optimizer.step()


	def eval_policy(self, replay_buffer, policy, batch_size=10000):
		state, action, next_state, reward, not_done = replay_buffer.sample_all()
		if not self.mujoco:
			action = torch.nn.functional.one_hot(action.to(torch.int64), num_classes=self.action_dim).to(torch.float32)
		return float(((self.W * self.encoder_decoder.latent(state, action)).mean(1,keepdim=True) * reward).mean())