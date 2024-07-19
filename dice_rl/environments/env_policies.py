# Copyright 2020 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import os
import logging
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union, Mapping

import numpy as np
import tensorflow.compat.v2 as tf

from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import actor_policy
from tf_agents.policies import greedy_policy
from tf_agents.policies import q_policy
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step
from tf_agents.utils import nest_utils
import tensorflow_probability as tfp
import tf_agents

from dice_rl.environments import suites
from dice_rl.environments.infinite_cartpole import InfiniteCartPole
from dice_rl.environments.infinite_frozenlake import InfiniteFrozenLake
#from dice_rl.environments.infinite_reacher import InfiniteReacher
from dice_rl.environments.gridworld import navigation
from dice_rl.environments.gridworld import maze
from dice_rl.environments.gridworld import point_maze
from dice_rl.environments.gridworld import taxi
from dice_rl.environments.gridworld import tree
from dice_rl.environments.gridworld import low_rank
from dice_rl.environments import bandit
from dice_rl.environments import bernoulli_bandit
from dice_rl.environments import line
from dice_rl.environments import contextual_bandit
import dice_rl.utils.common as common_lib



# Function to create TensorFlow QNetwork
def create_tf_q_network(tf_env, hidden_sizes):
    q_net = q_network.QNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        fc_layer_params=hidden_sizes
    )
    return q_net


def assign_weights_to_tf_model(tf_model, pytorch_weights, pytorch_activations):
    def get_tf_activation(name):
        if name == 'ReLU':
            return tf.keras.activations.relu
        elif name == 'Tanh':
            return tf.keras.activations.tanh
        elif name == 'Sigmoid':
            return tf.keras.activations.sigmoid
        elif name == 'LeakyReLU':
            return tf.keras.layers.LeakyReLU()
        else:
            return None

    def extract_dense_layers(encoding_network):
        dense_layers = []
        for layer in encoding_network.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                dense_layers.append(layer)
            elif hasattr(layer, 'layers'):
                dense_layers.extend(extract_dense_layers(layer))
        return dense_layers

    encoding_network_layers = []
    for layer in tf_model.layers:
        if isinstance(layer, tf_agents.networks.encoding_network.EncodingNetwork):
            encoding_network_layers.extend(extract_dense_layers(layer))
        elif isinstance(layer, tf.keras.layers.Dense):
            encoding_network_layers.append(layer)

    weight_index = 0
    for layer, activation in zip(encoding_network_layers, pytorch_activations):
        print(
            f"TF Layer: {layer.name}, Shape: {layer.get_weights()[0].shape if layer.get_weights() else 'Uninitialized'}")
        if weight_index < len(pytorch_weights):
            weight, bias = pytorch_weights[weight_index]
            weight_index += 1
            if hasattr(layer, 'kernel') and hasattr(layer, 'bias'):  # Ensure the layer has these attributes
                print(f"Assigning weights to layer {layer.name}")
                if not layer.built:
                    input_shape = (None, weight.shape[0])
                    layer.build(input_shape)
                layer.set_weights([tf.transpose(tf.convert_to_tensor(weight)), tf.convert_to_tensor(bias)])
                if activation:
                    tf_activation = get_tf_activation(activation)
                    if isinstance(tf_activation, tf.keras.layers.Layer):
                        # Replace the activation function in place if it's a layer
                        layer.activation = None  # Clear existing activation
                        tf_model.add(tf_activation)
                    else:
                        layer.activation = tf_activation


# Ensure you call the correct functions in your main workflow
def get_dqn_policy(tf_env, actor_weights=None, actor_activations=None, hidden_sizes=(100,)):
    q_net = create_tf_q_network(tf_env, hidden_sizes)

    dummy_input = tf.zeros([1] + list(tf_env.observation_spec().shape))
    q_net(dummy_input)

    def print_layer_details(layer):
        weights = layer.get_weights()
        print(f"Layer: {layer.name}, Type: {type(layer)}")
        if hasattr(layer, 'activation'):
            print(f"  Activation: {layer.activation}")
        if weights:
            print(f"  Weight Shape: {weights[0].shape}, First 5 Weights: {weights[0].flatten()[:5]}")
            if len(weights) > 1:
                print(f"  Bias Shape: {weights[1].shape}, First 5 Biases: {weights[1].flatten()[:5]}")

    print("TensorFlow QNetwork Layers:")
    for layer in q_net.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            print_layer_details(layer)
        elif isinstance(layer, tf_agents.networks.encoding_network.EncodingNetwork):
            for sub_layer in layer.layers:
                if isinstance(sub_layer, tf.keras.layers.Dense):
                    print_layer_details(sub_layer)

    if actor_weights and actor_activations:
        assign_weights_to_tf_model(q_net, actor_weights, actor_activations)

    print("After TensorFlow QNetwork Layers:")
    for layer in q_net.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            print_layer_details(layer)
        elif isinstance(layer, tf_agents.networks.encoding_network.EncodingNetwork):
            for sub_layer in layer.layers:
                if isinstance(sub_layer, tf.keras.layers.Dense):
                    print_layer_details(sub_layer)

    policy = q_policy.QPolicy(
        tf_env.time_step_spec(),
        action_spec=tf_env.action_spec(),
        q_network=q_net
    )
    return policy



def get_sac_policy(tf_env):
  actor_net = actor_distribution_network.ActorDistributionNetwork(
      tf_env.observation_spec(),
      tf_env.action_spec(),
      fc_layer_params=(256, 256),
      continuous_projection_net=tanh_normal_projection_network
      .TanhNormalProjectionNetwork)
  policy = actor_policy.ActorPolicy(
      time_step_spec=tf_env.time_step_spec(),
      action_spec=tf_env.action_spec(),
      actor_network=actor_net,
      training=False)
  return policy


def load_policy(policy, env_name, load_dir, ckpt_file=None):
  policy = greedy_policy.GreedyPolicy(policy)
  checkpoint = tf.train.Checkpoint(policy=policy)
  print("load_dir", load_dir)
  if ckpt_file is None:
    checkpoint_filename = tf.train.latest_checkpoint(load_dir)
    print('none')
  else:
    checkpoint_filename = os.path.join(load_dir, ckpt_file)
  print("checkpoint_filename", checkpoint_filename)
  print('Loading policy from %s.' % checkpoint_filename)
  checkpoint.restore(checkpoint_filename).assert_existing_objects_matched()
  # Unwrap greedy wrapper.
  return policy.wrapped_policy

# def load_custom_policy(env, tf_env, load_dir, ckpt_file=None):
#     observation_space = env.observation_space
#     action_space = env.action_space
#     ac_kwargs = dict(hidden_sizes=[64, 32])
#
#     print("load_dir", load_dir)
#
#     # Initialize the TensorFlow model
#     custom_policy = MLPActorCriticTF(observation_space, action_space, tf_env, **ac_kwargs)
#
#     # Create the model's weights by calling it with a dummy input
#     dummy_input = np.zeros((1, observation_space.shape[0]), dtype=np.float32)
#     custom_policy(dummy_input)
#
#     policy = greedy_policy.GreedyPolicy(custom_policy)
#     checkpoint = tf.train.Checkpoint(policy=policy)
#     print("load_dir", load_dir)
#     if ckpt_file is None:
#         checkpoint_filename = tf.train.latest_checkpoint(load_dir)
#         print('none')
#     else:
#         checkpoint_filename = os.path.join(load_dir, ckpt_file)
#     print("checkpoint_filename", checkpoint_filename)
#     print('Loading policy from %s.' % checkpoint_filename)
#     checkpoint.restore(checkpoint_filename).assert_existing_objects_matched()
#     # Unwrap greedy wrapper.
#     return policy.wrapped_policy

import os
import tensorflow as tf
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.policies import epsilon_greedy_policy




def load_custom_policy(env, tf_env, load_dir, ckpt_file=None):
    observation_space = env.observation_space
    action_space = env.action_space
    ac_kwargs = dict(hidden_sizes=[64, 32])

    print("load_dir", load_dir)

    # Initialize the TensorFlow model
    custom_policy = MLPActorCriticTF(observation_space, action_space, tf_env, **ac_kwargs)

    # Create the model's weights by calling it with a dummy input
    dummy_input = np.zeros((1, observation_space.shape[0]), dtype=np.float32)
    custom_policy(dummy_input)

    # Load the weights
    if ckpt_file is None:
        checkpoint_filename = tf.train.latest_checkpoint(load_dir)
        if checkpoint_filename is None:
            raise ValueError(f"No checkpoint found in directory {load_dir}.")
    else:
        checkpoint_filename = os.path.join(load_dir, ckpt_file)

    checkpoint_filename = load_dir
    print("checkpoint_filename", checkpoint_filename)
    print('Loading policy from %s.' % checkpoint_filename)

    # Ensure the checkpoint file exists
    if not os.path.exists(checkpoint_filename):
        raise ValueError(f"Checkpoint file {checkpoint_filename} does not exist.")

    custom_policy.load_weights(checkpoint_filename).expect_partial()

    return custom_policy

def debug_print_shapes(time_step):
    print("time_step.reward shape:", time_step.reward.shape)
    print("time_step.discount shape:", time_step.discount.shape)
    print("time_step.observation shape:", time_step.observation.shape)
    print("time_step.step_type shape:", time_step.step_type.shape)

def ensure_batch_dimension(time_step):
    # Add a batch dimension if it is missing
    if len(time_step.discount.shape) == 0:
        time_step = tf.nest.map_structure(lambda x: tf.expand_dims(x, 0), time_step)
    return time_step


class CustomTFPolicy(tf_policy.TFPolicy):
    def __init__(self, time_step_spec, action_spec, actor_critic_model):
        super(CustomTFPolicy, self).__init__(time_step_spec, action_spec)
        self._actor_critic_model = actor_critic_model

    def _distribution(self, time_step, policy_state):
        observation = time_step.observation
        action_distribution, _ = self._actor_critic_model(observation)
        return policy_step.PolicyStep(action_distribution, policy_state)

    def _action(self, time_step, policy_state, seed):
        distribution_step = self._distribution(time_step, policy_state)
        actions = distribution_step.action.sample(seed=seed)
        # Debugging: Log the dtype of the actions and action_spec
        print(f"Actions dtype: {actions.dtype}, Action Spec dtype: {self.action_spec().dtype}")

        # Ensure actions match the dtype of action_spec
        actions = tf.cast(actions, self.action_spec().dtype)

        # Debugging: Log the actions and their dtype after casting
        print(f"Actions after casting: {actions.numpy()}, dtype: {actions.dtype}")

        return policy_step.PolicyStep(actions, distribution_step.state, distribution_step.info)



##
import torch


def load_pytorch_model(path, env):
    import dice_rl.ppo.algo.core as core

    ac_kwargs = dict(hidden_sizes=[64, 32])
    ac = core.MLPActorCritic(env.observation_space, env.action_space, **ac_kwargs)
    checkpoint = torch.load(path)
    ac.load_state_dict(checkpoint['model_state_dict'])
    return ac


import torch
import torch.nn as nn


def extract_pytorch_weights(pytorch_model):
    actor_weights = []
    critic_weights = []
    actor_activations = []
    critic_activations = []

    def print_layer_details(layer, layer_name):
        activation = None
        if isinstance(layer, nn.Sequential):
            for sub_layer in layer:
                if isinstance(sub_layer, (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU)):
                    activation = type(sub_layer).__name__
        elif isinstance(layer, (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU)):
            activation = type(layer).__name__
        else:
            activation = 'None'

        print(f"  Layer: {layer_name}, Type: {type(layer)}, Activation: {activation}")
        if isinstance(layer, nn.Linear):
            print(
                f"    Weight Shape: {layer.weight.shape}, First 5 Weights: {layer.weight.detach().numpy().flatten()[:5]}")
            print(f"    Bias Shape: {layer.bias.shape}, First 5 Biases: {layer.bias.detach().numpy().flatten()[:5]}")

        return activation

    for module_name, module in pytorch_model.named_children():
        print(f"Module: {module_name}, Type: {type(module)}")

        if isinstance(module, (core.MLPGaussianActor, core.MLPCategoricalActor)):
            net = module.mu_net if hasattr(module, 'mu_net') else module.logits_net
            last_activation = 'None'
            for layer_name, layer in net.named_children():
                if isinstance(layer, nn.Linear):
                    weights = (layer.weight.detach().numpy(), layer.bias.detach().numpy())
                    actor_weights.append(weights)
                else:
                    last_activation = print_layer_details(layer, layer_name)
                    actor_activations.append(last_activation)  # Associate last activation with this layer
                print_layer_details(layer, layer_name)
            if isinstance(net[-1], nn.Linear):
                actor_activations[-1] = 'None'  # Last layer has no activation

        elif isinstance(module, core.MLPCritic):
            last_activation = 'None'
            for layer_name, layer in module.v_net.named_children():
                if isinstance(layer, nn.Linear):
                    weights = (layer.weight.detach().numpy(), layer.bias.detach().numpy())
                    critic_weights.append(weights)
                    critic_activations.append(last_activation)  # Associate last activation with this layer
                last_activation = print_layer_details(layer, layer_name)
            if isinstance(module.v_net[-1], nn.Linear):
                critic_activations[-1] = 'None'  # Last layer has no activation

        elif isinstance(module, NNGammaCritic):
            last_activation = 'None'
            for layer_name, layer in module.body.named_children():
                if isinstance(layer, nn.Linear):
                    weights = (layer.weight.detach().numpy(), layer.bias.detach().numpy())
                    critic_weights.append(weights)
                    critic_activations.append(last_activation)  # Associate last activation with this layer
                last_activation = print_layer_details(layer, layer_name)
            critic_weights.append((module.critic.weight.detach().numpy(), module.critic.bias.detach().numpy()))
            critic_activations.append('None')  # Assuming no activation for final layer
            print_layer_details(module.critic, 'critic')
            critic_weights.append((module.weight[0].weight.detach().numpy(), module.weight[0].bias.detach().numpy()))
            critic_activations.append('None')  # Assuming no activation for final layer
            print_layer_details(module.weight[0], 'weight[0]')

    return actor_weights, actor_activations, critic_weights, critic_activations


def get_env_and_custom_policy(env_name, pytorch_model_path, env_seed=0, epsilon=0.0, ckpt_file=None):
    gym_env = suite_gym.load(env_name)
    gym_env.seed(env_seed)
    env = tf_py_environment.TFPyEnvironment(gym_env)

    pytorch_model = load_pytorch_model(pytorch_model_path, gym_env)
    actor_weights, actor_activations, critic_weights, critic_activations = extract_pytorch_weights(pytorch_model)

    print("actor_activations", actor_activations)

    dqn_policy = get_dqn_policy(env, actor_weights, actor_activations, hidden_sizes=[64, 32])

    # Load the policy if a checkpoint is provided
    # policy = load_policy(dqn_policy, load_dir, ckpt_file)
    return env, EpsilonGreedyPolicy(dqn_policy, epsilon=epsilon, emit_log_probability=True)


def get_env_and_dqn_policy(env_name,
                           load_dir,
                           env_seed=0,
                           epsilon=0.0,
                           ckpt_file=None):
  gym_env = suites.load_gym(env_name)
  gym_env.seed(env_seed)
  env = tf_py_environment.TFPyEnvironment(gym_env)
  dqn_policy = get_dqn_policy(env)
  policy = load_policy(dqn_policy, env_name, load_dir, ckpt_file)
  return env, EpsilonGreedyPolicy(
      policy, epsilon=epsilon, emit_log_probability=True)


def get_env_and_policy(load_dir,
                       env_name,
                       alpha,
                       env_seed=0,
                       tabular_obs=False):
  if env_name == 'taxi':
    env = taxi.Taxi(tabular_obs=tabular_obs)
    env.seed(env_seed)
    policy_fn, policy_info_spec = taxi.get_taxi_policy(
        load_dir, env, alpha=alpha, py=False)
    tf_env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env))
    policy = common_lib.TFAgentsWrappedPolicy(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        policy_fn,
        policy_info_spec,
        emit_log_probability=True)
  elif env_name == 'grid':
    env = navigation.GridWalk(tabular_obs=tabular_obs)
    env.seed(env_seed)
    policy_fn, policy_info_spec = navigation.get_navigation_policy(
        env, epsilon_explore=0.1 + 0.6 * (1 - alpha), py=False)
    tf_env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env))
    policy = common_lib.TFAgentsWrappedPolicy(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        policy_fn,
        policy_info_spec,
        emit_log_probability=True)
  elif env_name == 'four_rooms':
    env = navigation.FourRooms(tabular_obs=tabular_obs)
    env.seed(env_seed)
    policy_fn, policy_info_spec = navigation.get_navigation_policy(
        env, epsilon_explore=0.1 + 0.6 * (1 - alpha), py=False)
    tf_env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env))
    policy = common_lib.TFAgentsWrappedPolicy(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        policy_fn,
        policy_info_spec,
        emit_log_probability=True)
  elif 'maze:' in env_name:
    # Format is in maze:<size>-<type>
    name, wall_type = env_name.split('-')
    size = int(name.split(':')[-1])
    env = maze.Maze(size, wall_type, maze_seed=env_seed)
    env.seed(env_seed)
    policy_fn, policy_info_spec = navigation.get_navigation_policy(
        env, epsilon_explore=1 - alpha, py=False)
    tf_env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env))
    policy = common_lib.TFAgentsWrappedPolicy(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        policy_fn,
        policy_info_spec,
        emit_log_probability=True)
  elif env_name == 'point_maze':
    env = point_maze.PointMaze(tabular_obs=tabular_obs)
    env.seed(env_seed)
    policy_fn, policy_info_spec = point_maze.get_navigation_policy(
        env, epsilon_explore=1. - alpha, py=False)
    tf_env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env))
    policy = common_lib.TFAgentsWrappedPolicy(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        policy_fn,
        policy_info_spec,
        emit_log_probability=True)
  elif env_name == 'low_rank':
    env = low_rank.LowRank()
    env.seed(env_seed)
    policy_fn, policy_info_spec = low_rank.get_low_rank_policy(
        env, epsilon_explore=0.1 + 0.8 * (1 - alpha), py=False)
    tf_env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env))
    policy = common_lib.TFAgentsWrappedPolicy(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        policy_fn,
        policy_info_spec,
        emit_log_probability=True)
  elif env_name == 'tree':
    env = tree.Tree(branching=2, depth=10)
    env.seed(env_seed)
    policy_fn, policy_info_spec = tree.get_tree_policy(
        env, epsilon_explore=0.1 + 0.8 * (1 - alpha), py=False)
    tf_env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env))
    policy = common_lib.TFAgentsWrappedPolicy(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        policy_fn,
        policy_info_spec,
        emit_log_probability=True)
  elif env_name == 'lowrank_tree':
    env = tree.Tree(branching=2, depth=3, duplicate=10)
    env.seed(env_seed)
    policy_fn, policy_info_spec = tree.get_tree_policy(
        env, epsilon_explore=0.1 + 0.8 * (1 - alpha), py=False)
    tf_env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env))
    policy = common_lib.TFAgentsWrappedPolicy(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        policy_fn,
        policy_info_spec,
        emit_log_probability=True)
  elif env_name == 'bernoulli_bandit':
    env = bernoulli_bandit.BernoulliBandit(num_arms=2)
    env.seed(env_seed)
    policy_fn, policy_info_spec = bernoulli_bandit.get_bandit_policy(
        env, epsilon_explore=1 - alpha, bernoulli_prob=alpha, py=False)
    tf_env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env))
    policy = common_lib.TFAgentsWrappedPolicy(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        policy_fn,
        policy_info_spec,
        emit_log_probability=True)
  elif env_name.startswith('bandit'):
    num_arms = int(env_name[6:]) if len(env_name) > 6 else 2
    env = bandit.Bandit(num_arms=num_arms)
    env.seed(env_seed)
    policy_fn, policy_info_spec = bandit.get_bandit_policy(
        env, epsilon_explore=1 - alpha, py=False)
    tf_env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env))
    policy = common_lib.TFAgentsWrappedPolicy(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        policy_fn,
        policy_info_spec,
        emit_log_probability=True)
  elif env_name.startswith('contextual_bandit'):
    num_arms = int(env_name[17:]) if len(env_name) > 17 else 2
    env = contextual_bandit.ContextualBandit(
        num_arms=num_arms, num_rewards=num_arms // 2)
    env.seed(env_seed)
    policy_fn, policy_info_spec = contextual_bandit.get_contextual_bandit_policy(
        env, epsilon_explore=1 - alpha, py=False)
    tf_env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env))
    policy = common_lib.TFAgentsWrappedPolicy(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        policy_fn,
        policy_info_spec,
        emit_log_probability=True)
  elif env_name == 'small_tree':
    env = tree.Tree(branching=2, depth=3, loop=True)
    env.seed(env_seed)
    policy_fn, policy_info_spec = tree.get_tree_policy(
        env, epsilon_explore=0.1 + 0.8 * (1 - alpha), py=False)
    tf_env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env))
    policy = common_lib.TFAgentsWrappedPolicy(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        policy_fn,
        policy_info_spec,
        emit_log_probability=True)
  elif env_name == 'CartPole-v1':
    epsilon = (1 - alpha)
    print("loading policy epsilon", epsilon)
    tf_env, policy = get_env_and_custom_policy(
        env_name,
        os.path.join(load_dir),
        env_seed=env_seed,
        epsilon=epsilon)
  elif env_name == 'cartpole':  # Infinite-horizon cartpole.
    tf_env, policy = get_env_and_dqn_policy(
        'CartPole-v0',
        os.path.join(load_dir, 'CartPole-v0-250', 'train', 'policy'),
        env_seed=env_seed,
        epsilon=0.3 + 0.15 * (1 - alpha))
    env = InfiniteCartPole()
    tf_env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env))
  elif env_name == 'FrozenLake-v1':
    tf_env, policy = get_env_and_dqn_policy(
        'FrozenLake-v1',
        os.path.join(load_dir, 'FrozenLake-v0', 'train', 'policy'),
        env_seed=env_seed,
        epsilon=0.2 * (1 - alpha),
        ckpt_file='ckpt-100000')
  elif env_name == 'frozenlake':  # Infinite-horizon frozenlake.
    tf_env, policy = get_env_and_dqn_policy(
        'FrozenLake-v1',
        os.path.join(load_dir, 'FrozenLake-v0', 'train', 'policy'),
        env_seed=env_seed,
        epsilon=0.2 * (1 - alpha),
        ckpt_file='ckpt-100000')
    env = InfiniteFrozenLake()
    tf_env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env))
  elif env_name in ['Reacher-v2', 'reacher']:
    if env_name == 'Reacher-v2':
      env = suites.load_mujoco(env_name)
    else:
      env = gym_wrapper.GymWrapper(InfiniteReacher())
    env.seed(env_seed)
    tf_env = tf_py_environment.TFPyEnvironment(env)
    sac_policy = get_sac_policy(tf_env)
    directory = os.path.join(load_dir, 'Reacher-v2', 'train', 'policy')
    policy = load_policy(sac_policy, env_name, directory)
    policy = GaussianPolicy(
        policy, 0.4 - 0.3 * alpha, emit_log_probability=True)
  elif env_name == 'HalfCheetah-v2':
    env = suites.load_mujoco(env_name)
    env.seed(env_seed)
    tf_env = tf_py_environment.TFPyEnvironment(env)
    sac_policy = get_sac_policy(tf_env)
    directory = os.path.join(load_dir, env_name, 'train', 'policy')
    policy = load_policy(sac_policy, env_name, directory)
    policy = GaussianPolicy(
        policy, 0.2 - 0.1 * alpha, emit_log_probability=True)
  elif env_name == 'line':
    tf_env, policy = line.get_line_env_and_policy(env_seed)
  else:
    raise ValueError('Unrecognized environment %s.' % env_name)

  return tf_env, policy


def get_target_policy(load_dir, env_name, tabular_obs, alpha=1.0):
  """Gets target policy."""
  print("loading target policy")
  tf_env, tf_policy = get_env_and_policy(
      load_dir, env_name, alpha, tabular_obs=tabular_obs)
  return tf_policy


class EpsilonGreedyPolicy(tf_policy.TFPolicy):
  """An epsilon-greedy policy that can return distributions."""

  def __init__(self, policy, epsilon, emit_log_probability=True):
    self._wrapped_policy = policy
    self._epsilon = epsilon
    if not common_lib.is_categorical_spec(policy.action_spec):
      raise ValueError('Action spec must be categorical to define '
                       'epsilon-greedy policy.')

    super(EpsilonGreedyPolicy, self).__init__(
        policy.time_step_spec,
        policy.action_spec,
        policy.policy_state_spec,
        policy.info_spec,
        emit_log_probability=emit_log_probability)

  @property
  def wrapped_policy(self):
    return self._wrapped_policy

  def _variables(self):
    return self._wrapped_policy.variables()

  def _get_epsilon(self):
    if callable(self._epsilon):
      return self._epsilon()
    else:
      return self._epsilon

  def _distribution(self, time_step, policy_state):
    batched = nest_utils.is_batched_nested_tensors(time_step,
                                                   self._time_step_spec)
    if not batched:
      time_step = nest_utils.batch_nested_tensors(time_step)

    policy_dist_step = self._wrapped_policy.distribution(
        time_step, policy_state)
    policy_state = policy_dist_step.state
    policy_info = policy_dist_step.info
    policy_logits = policy_dist_step.action.logits_parameter()
    action_size = tf.shape(policy_logits)[-1]

    greedy_probs = tf.one_hot(tf.argmax(policy_logits, -1), action_size)
    uniform_probs = (
        tf.ones(tf.shape(policy_logits)) / tf.cast(action_size, tf.float32))
    epsilon = self._get_epsilon()
    mixed_probs = (1 - epsilon) * greedy_probs + epsilon * uniform_probs
    if not batched:
      mixed_probs = tf.squeeze(mixed_probs, 0)
      policy_state = nest_utils.unbatch_nested_tensors(policy_state)
      policy_info = nest_utils.unbatch_nested_tensors(policy_info)
    mixed_dist = tfp.distributions.Categorical(
        probs=mixed_probs, dtype=policy_dist_step.action.dtype)

    return policy_step.PolicyStep(mixed_dist, policy_state, policy_info)


class GaussianPolicy(tf_policy.TFPolicy):
  """An gaussian policy that can return distributions."""

  def __init__(self, policy, scale, emit_log_probability=True):
    self._wrapped_policy = policy
    self._scale = scale

    super(GaussianPolicy, self).__init__(
        policy.time_step_spec,
        policy.action_spec,
        policy.policy_state_spec,
        policy.info_spec,
        emit_log_probability=emit_log_probability)

  @property
  def wrapped_policy(self):
    return self._wrapped_policy

  def _variables(self):
    return self._wrapped_policy.variables()

  def _get_epsilon(self):
    if callable(self._scale):
      return self._scale()
    else:
      return self._scale

  def _distribution(self, time_step, policy_state):
    batched = nest_utils.is_batched_nested_tensors(time_step,
                                                   self._time_step_spec)
    if not batched:
      time_step = nest_utils.batch_nested_tensors(time_step)

    policy_dist_step = self._wrapped_policy.distribution(
        time_step, policy_state)
    policy_state = policy_dist_step.state
    policy_mean_action = policy_dist_step.action.mean()
    policy_info = policy_dist_step.info

    if not batched:
      policy_state = nest_utils.unbatch_nested_tensors(policy_state)
      policy_mean_action = nest_utils.unbatch_nested_tensors(policy_mean_action)
      policy_info = nest_utils.unbatch_nested_tensors(policy_info)

    gaussian_dist = tfp.distributions.MultivariateNormalDiag(
          loc=policy_mean_action,
          scale_diag=tf.ones_like(policy_mean_action) * self._scale)

    return policy_step.PolicyStep(gaussian_dist, policy_state,
                                  policy_info)


def get_env_and_policy_from_weights(env_name: str,
                                    weights: Mapping[str, np.ndarray],
                                    n_hidden: int = 300,
                                    min_log_std: float = -5,
                                    max_log_std: float = 2):
  """Return tf_env and policy from dictionary of weights.

  Assumes that the policy has 2 hidden layers with 300 units, ReLu activations,
  and outputs a normal distribution squashed by a Tanh.

  Args:
    env_name: Name of the environment.
    weights: Dictionary of weights containing keys: fc0/weight, fc0/bias,
      fc0/weight, fc0/bias, last_fc/weight, last_fc_log_std/weight,
      last_fc/bias, last_fc_log_std/bias

  Returns:
    tf_env: TF wrapped env.
    policy: TF Agents policy.
  """
  env = suites.load_mujoco(env_name)
  tf_env = tf_py_environment.TFPyEnvironment(env)
  std_transform = (
      lambda x: tf.exp(tf.clip_by_value(x, min_log_std, max_log_std)))
  actor_net = actor_distribution_network.ActorDistributionNetwork(
      tf_env.observation_spec(),
      tf_env.action_spec(),
      fc_layer_params=(n_hidden, n_hidden),
      continuous_projection_net=functools.partial(
          tanh_normal_projection_network.TanhNormalProjectionNetwork,
          std_transform=std_transform),
      activation_fn=tf.keras.activations.relu,
  )
  policy = actor_policy.ActorPolicy(
      time_step_spec=tf_env.time_step_spec(),
      action_spec=tf_env.action_spec(),
      actor_network=actor_net,
      training=False)

  # Set weights
  actor_net._encoder.layers[1].set_weights(  # pylint: disable=protected-access
      [weights['fc0/weight'].T, weights['fc0/bias']])
  actor_net._encoder.layers[2].set_weights(  # pylint: disable=protected-access
      [weights['fc1/weight'].T, weights['fc1/bias']])
  actor_net._projection_networks.layers[0].set_weights(  # pylint: disable=protected-access
      [
          np.concatenate(
              (weights['last_fc/weight'], weights['last_fc_log_std/weight']),
              axis=0).T,
          np.concatenate(
              (weights['last_fc/bias'], weights['last_fc_log_std/bias']),
              axis=0)
      ])
  return tf_env, policy


import dice_rl.ppo.algo.core as core
import torch

# load target policy
def load(path,env):
    ac_kwargs = dict(hidden_sizes=[64,32])

    ac = core.MLPActorCritic(env.observation_space, env.action_space, **ac_kwargs)
    checkpoint = torch.load(path)
    ac.load_state_dict(checkpoint['model_state_dict'])
    return ac

import tensorflow as tf
import tensorflow_probability as tfp
from gym.spaces import Box, Discrete

tfd = tfp.distributions

class MLPActorCriticTF(tf.keras.Model):
    def __init__(self, observation_space, action_space, tf_env, emit_log_probability=True, hidden_sizes=(64, 64), activation='tanh', ):
        super(MLPActorCriticTF, self).__init__()
        obs_dim = observation_space.shape[0]

        self.time_step_spec = tf_env.time_step_spec()
        self.action_spec = tf_env.action_spec()
        self.policy_state_spec = tf_env.observation_spec()
        self.info_spec = {}
        self.emit_log_probability = emit_log_probability

        if isinstance(action_space, Box):
            self.pi = MLPGaussianActorTF(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActorTF(obs_dim, action_space.n, hidden_sizes, activation)

        self.v = MLPCriticTF(obs_dim, hidden_sizes, activation)

    def call(self, obs):
        pi_dist = self.pi._distribution(obs)
        v = self.v(obs)
        return pi_dist, v

class MLPGaussianActorTF(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super(MLPGaussianActorTF, self).__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = tf.Variable(log_std, trainable=True)
        self.mu_net = self._build_mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _build_mlp(self, sizes, activation):
        model = tf.keras.Sequential()
        for j in range(len(sizes) - 1):
            model.add(tf.keras.layers.Dense(sizes[j + 1], activation=activation if j < len(sizes) - 2 else None))
        return model

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = tf.exp(self.log_std)
        return tfd.Normal(mu, std)

class MLPCategoricalActorTF(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super(MLPCategoricalActorTF, self).__init__()
        self.logits_net = self._build_mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _build_mlp(self, sizes, activation):
        model = tf.keras.Sequential()
        for j in range(len(sizes) - 1):
            model.add(tf.keras.layers.Dense(sizes[j + 1], activation=activation if j < len(sizes) - 2 else None))
        return model

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return tfd.Categorical(logits=logits)

class MLPCriticTF(tf.keras.Model):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super(MLPCriticTF, self).__init__()
        self.v_net = self._build_mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def _build_mlp(self, sizes, activation):
        model = tf.keras.Sequential()
        for j in range(len(sizes) - 1):
            model.add(tf.keras.layers.Dense(sizes[j + 1], activation=activation if j < len(sizes) - 2 else None))
        return model

    def call(self, obs):
        return tf.squeeze(self.v_net(obs), axis=-1)


# Convert PyTorch model to TensorFlow model and save
def load_tf_model_from_pytorch(pytorch_model, observation_space, action_space, tf_env, hidden_sizes=(64, 32), save_dir='tf_model'):
    ac_tf = MLPActorCriticTF(observation_space, action_space, tf_env, True, hidden_sizes)

    # Create the model's weights by calling it with a dummy input
    dummy_input = np.zeros((1, observation_space.shape[0]), dtype=np.float32)
    ac_tf(dummy_input)

    # Transfer parameters from PyTorch to TensorFlow
    for layer_tf, layer_torch in zip(ac_tf.trainable_variables, pytorch_model.parameters()):
        if len(layer_tf.shape) == 2:  # Dense layer weights
            layer_tf.assign(tf.transpose(layer_torch.detach().numpy()))
        else:  # Biases and other parameters
            layer_tf.assign(layer_torch.detach().numpy())

    # Save the TensorFlow model
    ac_tf.save_weights(save_dir)
    return ac_tf


def main():
    import gym
    path = './exper/cartpole.pth'
    env_name = 'CartPole-v0'
    env = gym.make(env_name)

    gym_env = suite_gym.load(env_name)
    tf_env = tf_py_environment.TFPyEnvironment(gym_env)

    save_dir = './tf_model/cartpole/'
    ac = load(path,env)
    load_tf_model_from_pytorch(ac, env.observation_space, env.action_space, tf_env, hidden_sizes=(64, 32), save_dir=save_dir)

if __name__ == '__main__':
    main()
