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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import numpy as np
import os
import tensorflow.compat.v2 as tf
tf.compat.v1.enable_v2_behavior()
import pickle

from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment


import dice_rl.environments.env_policies as env_policies
import dice_rl.data.tf_agents_onpolicy_dataset as tf_agents_onpolicy_dataset
import dice_rl.estimators.estimator as estimator_lib
import dice_rl.utils.common as common_utils
from dice_rl.data.dataset import Dataset, EnvStep, StepType
from dice_rl.data.tf_offpolicy_dataset import TFOffpolicyDataset
import dice_rl.ppo.algo.core as core
import torch
from torch.distributions import Normal
import gymnasium as gym
from tf_agents.environments import gym_wrapper

from tf_agents.specs import tensor_spec
from dice_rl.environments.env_policies import load, convert_to_gym_observation_space


# # load target policy
# def load(path,env):
#     ac_kwargs = dict(hidden_sizes=[64,32])
#     ac = core.MLPActorCritic(convert_to_gym_observation_space(env.observation_space),
#                              convert_to_gym_observation_space(env.action_space), **ac_kwargs)
#     checkpoint = torch.load(path)
#     ac.load_state_dict(checkpoint['model_state_dict'])
#     return ac

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, fold, gamma=0.99):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.tim_buf = np.zeros(size, dtype=np.int32)
        self.logtarg_buf = np.zeros(size, dtype=np.float32)
        self.prod_buf = np.zeros(size, dtype=np.float32)
        self.logbev_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.fold = fold
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, next_obs,act, rew, tim, logbev, logtarg):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.tim_buf[self.ptr] = tim
        self.logbev_buf[self.ptr] = logbev
        self.logtarg_buf[self.ptr] = logtarg
        self.ptr += 1

    def finish_path(self):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = self.logtarg_buf - self.logbev_buf
        self.prod_buf[path_slice] = np.append(0,core.discount_cumsum(deltas[path_slice], 1)[:-1])

        self.path_start_idx = self.ptr

    def sample(self,batch_size,fold_num):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        interval = int(self.ptr / self.fold)
        if self.fold>1:
            ind = np.random.randint(self.ptr-interval, size=batch_size)
            ind = ind + np.where(ind >= fold_num * interval, 1, 0) * interval
        else:
            ind = np.random.randint(self.ptr, size=batch_size)

        data = dict(obs=self.obs_buf[ind], act=self.act_buf[ind],
                    prod=self.prod_buf[ind],tim=self.tim_buf[ind],
                    logbev=self.logbev_buf[ind], logtarg=self.logtarg_buf[ind])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

    def delete_last_traj(self):
        self.ptr =self.path_start_idx

def collect_dataset(env_name,gamma,buffer_size=20,max_len=200,
                    path='./exper/cartpole_998.pth', random_weight=0.2,fold=1,mujoco=False, seed=0, tf_dataset=None):
    env = gymnasium.make(env_name)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    env.reset(seed=seed)

    ac = load(path, env)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    buf = PPOBuffer(obs_dim, act_dim, buffer_size * max_len, fold, gamma)

    # Initialize the cache to store steps from one trajectory
    trajectory_cache = []

    o, _ = env.reset()
    num_traj, ep_len = 0, 0

    if isinstance(env.action_space, gymnasium.spaces.Box):
        action_range = env.action_space.high - env.action_space.low
        assert np.any(action_range > 0)
        unif = 1 / np.prod(action_range)
    elif isinstance(env.action_space, gymnasium.spaces.Discrete):
        unif = 1 / env.action_space.n

    while num_traj < buffer_size:
        pi = ac.pi._distribution(torch.as_tensor(o, dtype=torch.float32))
        if mujoco:
            std = torch.exp(ac.pi.log_std)
            mu = ac.pi.mu_net(torch.as_tensor(o, dtype=torch.float32))
            beh_pi = Normal(mu, std * random_weight)
            a = beh_pi.sample().numpy()
            logtarg = pi.log_prob(torch.as_tensor(a)).sum(axis=-1).detach().numpy()
            logbev = beh_pi.log_prob(torch.as_tensor(a)).sum(axis=-1).detach().numpy()
        else:
            targ_a, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
            if np.random.random() < random_weight:
                a = env.action_space.sample()
            else:
                a = targ_a
            logtarg = ac.pi._log_prob_from_distribution(pi, torch.as_tensor(a)).detach().numpy()
            logbev = np.log(random_weight * unif + (1 - random_weight) * np.exp(logtarg))

        next_o, r, d, truncated, _ = env.step(a)
        ep_len += 1

        # Store each step in the buffer and also cache the steps for the current trajectory
        buf.store(o, next_o, a, r, ep_len - 1, logbev, logtarg)

        # Cache the trajectory steps
        trajectory_cache.append({
            'obs': o,
            'act': a,
            'rew': r,
            'next_obs': next_o,
            'done': d,
            'step_num': ep_len - 1
        })

        # Update obs
        o = next_o

        terminal = d or truncated
        epoch_ended = ep_len == max_len

        if terminal or epoch_ended:
            # If terminal and not (epoch ended), delete trajectory and reset
            if terminal and not (epoch_ended):
                buf.delete_last_traj()
                o, _ = env.reset()
                ep_len, ep_ret, ep_avg_ret = 0, 0, 0
                trajectory_cache = []  # Reset cache as the trajectory is invalid
                continue

            # Finish the path

            # Add the cached trajectory steps to the TF dataset
            if tf_dataset is not None:
                for i, step in enumerate(trajectory_cache):
                    step_type = StepType.FIRST if i == 0 else StepType.MID
                    if i == len(trajectory_cache) - 1:
                        step_type = StepType.LAST

                    if isinstance(env.action_space, gymnasium.spaces.Box):
                        tf_action = tf.convert_to_tensor(step['act'], dtype=tf.float32)
                    elif isinstance(env.action_space, gymnasium.spaces.Discrete):
                        tf_action = tf.convert_to_tensor(step['act'], dtype=tf.int32)
                    else:
                        raise ValueError("Unsupported action space type")

                    # Create the EnvStep using cached step data
                    env_step = EnvStep(
                        step_type=tf.convert_to_tensor(step_type, dtype=tf.int32),
                        step_num=tf.convert_to_tensor(step['step_num'], dtype=tf.int64),
                        observation=tf.convert_to_tensor(step['obs'], dtype=tf.float32),
                        action=tf_action,
                        reward=tf.convert_to_tensor(step['rew'], dtype=tf.float32),
                        discount=tf.convert_to_tensor(gamma, dtype=tf.float32),
                        policy_info=tf.convert_to_tensor(logtarg, dtype=tf.float32),
                        env_info=tf.convert_to_tensor(0, dtype=tf.int32),
                        other_info=tf.convert_to_tensor(0, dtype=tf.int32)
                    )
                    tf_dataset.add_step(env_step)

            # Reset the cache and environment
            trajectory_cache = []
            o, _ = env.reset()
            ep_len, ep_ret, ep_avg_ret = 0, 0, 0
            num_traj += 1
            buf.finish_path()


    return buf


def get_onpolicy_dataset(load_dir, env_name, tabular_obs, max_trajectory_length,
                         alpha, seed):
  """Get on-policy dataset."""
  tf_env, tf_policy = env_policies.get_env_and_policy(
      load_dir, env_name, alpha, env_seed=seed, tabular_obs=tabular_obs)

  dataset = tf_agents_onpolicy_dataset.TFAgentsOnpolicyDataset(
      tf_env, tf_policy,
      episode_step_limit=max_trajectory_length)
  return dataset


def add_episodes_to_dataset(episodes, valid_ids, write_dataset):
  num_episodes = 1 if tf.rank(valid_ids) == 1 else tf.shape(valid_ids)[0]
  for ep_id in range(num_episodes):
    if tf.rank(valid_ids) == 1:
      this_valid_ids = valid_ids
      this_episode = episodes
    else:
      this_valid_ids = valid_ids[ep_id, ...]
      this_episode = tf.nest.map_structure(
          lambda t: t[ep_id, ...], episodes)

    episode_length = tf.shape(this_valid_ids)[0]
    for step_id in range(episode_length):
      this_valid_id = this_valid_ids[step_id]
      this_step = tf.nest.map_structure(
          lambda t: t[step_id, ...], this_episode)
      if this_valid_id:
        write_dataset.add_step(this_step)


def create_env_step_spec_from_gym(env_name):
    """Create the EnvStep spec from Gymnasium environment."""
    env = gymnasium.make(env_name)
    obs_space = env.observation_space
    act_space = env.action_space

    # Define TensorSpecs from Gymnasium space
    if isinstance(obs_space, gymnasium.spaces.Box):
        obs_spec = tensor_spec.TensorSpec(shape=obs_space.shape, dtype=np.float32, name='observation')
    elif isinstance(obs_space, gymnasium.spaces.Discrete):
        obs_spec = tensor_spec.TensorSpec(shape=[], dtype=tf.int32, name='observation')
    else:
        raise ValueError("Unsupported observation space type")

    if isinstance(act_space, gymnasium.spaces.Box):
        act_spec = tensor_spec.TensorSpec(shape=act_space.shape, dtype=np.float32, name='action')
    elif isinstance(act_space, gymnasium.spaces.Discrete):
        act_spec = tensor_spec.BoundedTensorSpec(shape=(), dtype=tf.int32,
                                                 name='action', minimum=array(0, dtype=int32),
                                                 maximum=array(act_space.n, dtype=int32))

    else:
        raise ValueError("Unsupported action space type")

    reward_spec = tensor_spec.TensorSpec(shape=[], dtype=tf.float32, name='reward')
    step_type_spec = tensor_spec.TensorSpec(shape=[], dtype=tf.int32, name='step_type')
    discount_spec = tensor_spec.TensorSpec(shape=[], dtype=tf.float32, name='discount')

    # Add missing specs for policy_info, env_info, and other_info
    policy_info_spec = {'log_probability':tensor_spec.TensorSpec(shape=[], dtype=tf.float32, name='data_policy_info')}
    env_info_spec = tensor_spec.TensorSpec(shape=[], dtype=tf.int32, name='env_info')

    other_info_spec = tensor_spec.TensorSpec(shape=[], dtype=tf.int32, name='other_info')

    # Return the complete EnvStep spec
    return EnvStep(
        step_type=step_type_spec,
        step_num=tensor_spec.TensorSpec([], dtype=tf.int64, name='step_num'),
        observation=obs_spec,
        action=act_spec,
        reward=reward_spec,
        discount=discount_spec,
        policy_info=policy_info_spec,
        env_info=env_info_spec,
        other_info=other_info_spec
    )



def main(argv):
  env_name = FLAGS.env_name
  seed = FLAGS.seed
  tabular_obs = FLAGS.tabular_obs
  num_trajectory = FLAGS.num_trajectory
  max_trajectory_length = FLAGS.max_trajectory_length
  alpha = FLAGS.alpha
  save_dir = FLAGS.save_dir
  load_dir = FLAGS.load_dir
  force = FLAGS.force
  gamma = FLAGS.gamma
  random_weight =FLAGS.random_weight
  
  hparam_str = ('{ENV_NAME}_tabular{TAB}_alpha{ALPHA}_seed{SEED}_'
                'numtraj{NUM_TRAJ}_maxtraj{MAX_TRAJ}_gamma{GAMMA}_random{RANDOM_WEIGHT}').format(
      ENV_NAME=env_name,
      TAB=tabular_obs,
      ALPHA=alpha,
      SEED=seed,
      NUM_TRAJ=num_trajectory,
      MAX_TRAJ=max_trajectory_length,
      GAMMA=gamma,
      RANDOM_WEIGHT=random_weight)
  directory = os.path.join(save_dir, hparam_str)
  if tf.io.gfile.isdir(directory) and not force:
    raise ValueError('Directory %s already exists. Use --force to overwrite.' %
                     directory)

  np.random.seed(seed)
  tf.random.set_seed(seed)

  env_step_spec = create_env_step_spec_from_gym(env_name)
  write_dataset = TFOffpolicyDataset(
      env_step_spec,
      capacity=num_trajectory * (max_trajectory_length + 1))

  # Step 1: Collect data using the PyTorch model
  if env_name in ['CartPole-v1']:
      buffer = collect_dataset(env_name, path=load_dir, buffer_size=num_trajectory, max_len=max_trajectory_length, gamma=gamma,
                               random_weight=1 - alpha, fold=1, mujoco=False, seed=seed, tf_dataset=write_dataset)
  elif env_name in ['Hopper-v4','HalfCheetah-v4','Ant-v4','Walker2d-v4']:
      buffer = collect_dataset(env_name, path=load_dir, buffer_size=num_trajectory, max_len=max_trajectory_length, gamma=gamma,
                               random_weight=1 - alpha, fold=1, mujoco=True, seed=seed, tf_dataset=write_dataset)
  else:
      print("error")


  # Step 2: Convert the collected data to TensorFlow dataset
  # dataset = convert_to_tf_dataset(buffer)

  #dataset = get_onpolicy_dataset(load_dir, env_name, tabular_obs,
                                 # max_trajectory_length, alpha, seed)




  # batch_size = 20
  # for batch_num in range(1 + (num_trajectory - 1) // batch_size):
  #   num_trajectory_after_batch = min(num_trajectory, batch_size * (batch_num + 1))
  #   num_trajectory_to_get = num_trajectory_after_batch - batch_num * batch_size
  #   episodes, valid_steps = dataset.get_episode(
  #       batch_size=num_trajectory_to_get)
  #   add_episodes_to_dataset(episodes, valid_steps, write_dataset)
  #
  #   print('num episodes collected: %d', write_dataset.num_total_episodes)
  #   print('num steps collected: %d', write_dataset.num_steps)
  #
  #   estimate = estimator_lib.get_fullbatch_average(write_dataset)
  #   print('per step avg on offpolicy data', estimate)
  #   estimate = estimator_lib.get_fullbatch_average(write_dataset,
  #                                                  by_steps=False)
  #   print('per episode avg on offpolicy data', estimate)

  print('Saving dataset to %s.' % directory)
  if not tf.io.gfile.isdir(directory):
    tf.io.gfile.makedirs(directory)
  write_dataset.save_off(directory)

  load_env_step_spec = create_env_step_spec_from_gym(env_name)
  new_dataset = TFOffpolicyDataset(
      load_env_step_spec,
      capacity=num_trajectory * (max_trajectory_length + 1))

  print('Loading dataset.')
  new_dataset = TFOffpolicyDataset.load_off(directory)
  print('num loaded steps', new_dataset.num_steps)
  print('num loaded total steps', new_dataset.num_total_steps)
  print('num loaded episodes', new_dataset.num_episodes)
  print('num loaded total episodes', new_dataset.num_total_episodes)

  estimate = estimator_lib.get_fullbatch_average(new_dataset)
  print('per step avg on saved and loaded offpolicy data', estimate)
  estimate = estimator_lib.get_fullbatch_average(new_dataset,
                                                 by_steps=False)
  print('per episode avg on saved and loaded offpolicy data', estimate)

  print('Done!')


if __name__ == '__main__':
    FLAGS = flags.FLAGS

    flags.DEFINE_string('env_name', 'taxi', 'Environment name.')
    flags.DEFINE_integer('seed', 0, 'Initial random seed.')
    flags.DEFINE_integer('num_trajectory', 100,
                         'Number of trajectories to collect.')
    flags.DEFINE_integer('max_trajectory_length', 500,
                         'Cutoff trajectory at this step.')
    flags.DEFINE_float('alpha', 1.0,
                       'How close to target policy.')
    flags.DEFINE_bool('tabular_obs', True,
                      'Whether to use tabular observations.')
    flags.DEFINE_string('save_dir', None, 'Directory to save dataset to.')
    flags.DEFINE_string('load_dir', None, 'Directory to load policies from.')
    flags.DEFINE_bool('force', False,
                      'Whether to force overwriting any existing dataset.')
    flags.DEFINE_float('gamma', 0.99, 'Discount factor.')
    flags.DEFINE_float('random_weight', 0.2,
                       'random policy')
    app.run(main)
