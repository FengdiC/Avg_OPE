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

import collections
import numpy as np
import tensorflow.compat.v2 as tf

from typing import Any, Callable, List, Optional, Text, Tuple, Union

from tf_agents.replay_buffers import table
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
import tensorflow_probability as tfp

from dice_rl.data.dataset import EnvStep, OffpolicyDataset, StepType

import os
import os
import json
import pickle

EpisodeInfo = collections.namedtuple(
    'EpisodeInfo', ['episode_start_id', 'episode_end_id',
                    'episode_start_type', 'episode_end_type'])


class TFOffpolicyDataset(tf.Module, OffpolicyDataset):
  """An off-policy dataset based on TF-Agents' uniform replay buffer."""

  def __init__(self,
               spec: EnvStep,
               capacity: int = 1000000,
               name: Text = 'TFOffpolicyDataset',
               device: Text = 'cpu:*',
               table_fn=table.Table):
    """Creates a TFOffpolicyDataset.

    Args:
      data_spec: A spec determining the format of steps stored in this dataset.
      capacity: The capacity of the dataset.
      name: Name to place ops under.
      device: A TensorFlow device to place the Variables and ops.
      table_fn: Function to create tables `table_fn(data_spec, capacity)` that
        can read/write nested tensors.
    """
    super(TFOffpolicyDataset, self).__init__(name=name)
    self._spec = spec
    self._capacity_value = np.int64(capacity)

    self.env_capacity = capacity
    self.env_step_spec = spec

    self._episode_info_spec = EpisodeInfo(
        tensor_spec.TensorSpec([], dtype=tf.int64, name='episode_start_id'),
        tensor_spec.TensorSpec([], dtype=tf.int64, name='episode_end_id'),
        tensor_spec.TensorSpec([], dtype=tf.int32, name='episode_start_type'),
        tensor_spec.TensorSpec([], dtype=tf.int32, name='episode_end_type'))
    self._valid_steps_spec = tensor_spec.TensorSpec([], dtype=tf.int64,
                                                    name='valid_steps')

    self._name = name
    self._device = device
    self._table_fn = table_fn
    self._last_rows_read = None

    with tf.device(self._device), self.name_scope:
      self._capacity = tf.constant(capacity, dtype=tf.int64)
      self._data_table = table_fn(self._spec, self._capacity_value)
      self._episode_info_table = table_fn(self._episode_info_spec,
                                          self._capacity_value)
      self._valid_steps_table = table_fn(self._valid_steps_spec,
                                         self._capacity_value)
      self._last_step_id = tf.Variable(-1, name='last_step_id', dtype=tf.int64)
      self._last_episode_id = tf.Variable(-1, name='last_episode_id', dtype=tf.int64)
      self._last_valid_steps_id = tf.Variable(
          -1, name='last_valid_steps_id', dtype=tf.int64)

  def variables(self):
    return (self._data_table.variables() +
            self._episode_info_table.variables() +
            self._last_valid_steps_table.variables() +
            [self._last_step_id, self._last_episode_id,
             self._last_valid_steps_id])



  def save_constructor_info(self, directory):
    CONSTRUCTOR_PREFIX = 'dataset-ctr.pkl'
    import os
    import pickle

    constructor_info = {
      'type': self.__class__,
      'args': self.constructor_args_and_kwargs[0],
      'kwargs': self.constructor_args_and_kwargs[1]
    }
    with open(os.path.join(directory, CONSTRUCTOR_PREFIX), 'wb') as f:
      pickle.dump(constructor_info, f)

  def save_checkpoint(self, directory):
    CHECKPOINT_PREFIX = 'ckpt'
    import os
    checkpoint = tf.train.Checkpoint(dataset=self)
    checkpoint.save(file_prefix=os.path.join(directory, CHECKPOINT_PREFIX))

  @property
  def spec(self):
    # TF wraps EnvStep in a TupleWrapper. We need to put it back as an EnvStep.
    return EnvStep(*self._spec)

  @property
  def num_steps(self) -> Union[int, tf.Tensor]:
    return self._last_valid_steps_id + 1

  @property
  def num_total_steps(self) -> Union[int, tf.Tensor]:
    return self._last_step_id + 1

  @property
  def num_episodes(self) -> Union[int, tf.Tensor]:
    all_episode_infos = self._episode_info_table.read(
        tf.range(self._last_episode_id + 1))
    full_episodes = tf.logical_and(
        StepType.is_first(all_episode_infos.episode_start_type),
        StepType.is_last(all_episode_infos.episode_end_type))
    return tf.cast(
        tf.reduce_sum(tf.cast(full_episodes, tf.float32)),
        tf.int64)

  @property
  def num_total_episodes(self) -> Union[int, tf.Tensor]:
    return self._last_episode_id + 1

  @property
  def constructor_args_and_kwargs(self):
    args = [self.spec]
    kwargs = {'capacity': self._capacity,
              'name': self._name,
              'device': self._device,
              'table_fn': self._table_fn}
    return args, kwargs

  @property
  def last_rows_read(self):
    return self._last_rows_read

  @property
  def capacity(self):
    return self._capacity_value

  @property
  def device(self):
    return self._device

  @tf.Module.with_name_scope
  def add_step(self, env_step: EnvStep):
    tf.nest.assert_same_structure(env_step, self._spec)

    with tf.device(self._device):
      if self._last_step_id >= self._capacity - 1:
        #TODO(ofirnachum): implement circular dataset.
        raise ValueError('Dataset is over capacity.')
      self._last_step_id.assign_add(1)

      if StepType.is_first(env_step.step_type):
        # New episode, increment episode id;
        self._last_episode_id.assign_add(1)

      if self._last_episode_id < 0:
        raise ValueError('First added step must have type StepType.FIRST.')

      current_episode_info = self._episode_info_table.read(
          self._last_episode_id)

      if StepType.is_first(env_step.step_type):
        # Full episode is just this single step.
        current_episode_info = EpisodeInfo(
            self._last_step_id, self._last_step_id,
            env_step.step_type, env_step.step_type)
      else:
        # Update current episode with latest step as the 'end' step.
        current_episode_info = EpisodeInfo(
            current_episode_info.episode_start_id,
            self._last_step_id,
            current_episode_info.episode_start_type,
            env_step.step_type)

      write_episode_op = self._episode_info_table.write(
          self._last_episode_id, current_episode_info)
      write_step_op = self._data_table.write(
          self._last_step_id, env_step)
      ret_op =  tf.group(write_episode_op, write_step_op)

      if not StepType.is_last(env_step.step_type):
        # This new step is valid for sampling.
        self._last_valid_steps_id.assign_add(1)
        write_valid_steps_op = self._valid_steps_table.write(
            self._last_valid_steps_id, self._last_step_id)
        ret_op = tf.group(ret_op, write_valid_steps_op)

      return ret_op

  @tf.Module.with_name_scope
  def get_step(self, batch_size: Optional[int] = None,
               num_steps: Optional[int] = None) -> EnvStep:
    batch_size_ = batch_size
    if batch_size_ is None:
      batch_size_ = 1
    num_steps_ = num_steps
    if num_steps_ is None:
      num_steps_ = 1

    if self._last_valid_steps_id < 0:
      raise ValueError('No valid steps for sampling in the dataset.')
    all_valid_steps = self._valid_steps_table.read(
        tf.range(self._last_valid_steps_id + 1))
    # Can't collect trajectories that trail off end of dataset.
    if tf.reduce_min(all_valid_steps) + num_steps_ > self._last_step_id + 1:
      raise ValueError('Not enough steps in the dataset.')

    probs = tf.cast(all_valid_steps + num_steps_ <= self._last_step_id + 1,
                    tf.float32)
    probs /= tf.reduce_sum(probs)
    distribution = tfp.distributions.Categorical(probs=probs, dtype=tf.int64)
    sampled_valid_ids = distribution.sample(batch_size_)
    sampled_valid_steps = tf.gather(all_valid_steps, sampled_valid_ids)

    rows_to_get = (sampled_valid_steps[:, None] +
                   tf.range(num_steps_, dtype=tf.int64)[None, :])
    rows_to_get = tf.math.mod(rows_to_get, self._last_step_id + 1)
    steps = self._data_table.read(rows_to_get)
    self._last_rows_read = rows_to_get

    if num_steps is None:
      steps = tf.nest.map_structure(lambda t: tf.squeeze(t, 1), steps)
      self._last_rows_read = tf.squeeze(self._last_rows_read, 1)
    if batch_size is None:
      steps = tf.nest.map_structure(lambda t: tf.squeeze(t, 0), steps)
      self._last_rows_read = tf.squeeze(self._last_rows_read, 0)

    return steps

  def _get_episodes(self, episode_ids, truncate_episode_at):
    # Determine number of steps to return for each episode.
    episode_infos = self._episode_info_table.read(episode_ids)
    episode_lengths = 1 + tf.maximum(
        0, episode_infos.episode_end_id - episode_infos.episode_start_id)
    num_steps = tf.reduce_max(episode_lengths)
    if truncate_episode_at is not None:
      num_steps = tf.minimum(num_steps, truncate_episode_at)

    rows_to_get = (episode_infos.episode_start_id[:, None] +
                   tf.range(num_steps, dtype=tf.int64)[None, :])
    rows_to_get = tf.math.mod(rows_to_get, self._last_step_id + 1)
    steps = self._data_table.read(rows_to_get)
    self._last_rows_read = rows_to_get
    valid_steps = (tf.range(num_steps, dtype=tf.int64)[None, :] <
                   episode_lengths[:, None])
    return steps, valid_steps

  def _get_episodes_reward(self, episode_ids, truncate_episode_at=None):
    # Determine number of steps to return for each episode.
    episode_infos = self._episode_info_table.read(episode_ids)
    episode_lengths = 1 + tf.maximum(
      0, episode_infos.episode_end_id - episode_infos.episode_start_id)
    num_steps = tf.reduce_max(episode_lengths)
    if truncate_episode_at is not None:
      num_steps = tf.minimum(num_steps, truncate_episode_at)

    rows_to_get = (episode_infos.episode_start_id[:, None] +
                   tf.range(num_steps, dtype=tf.int64)[None, :])
    rows_to_get = tf.math.mod(rows_to_get, self._last_step_id + 1)
    steps = self._data_table.read(rows_to_get)
    self._last_rows_read = rows_to_get
    valid_steps = (tf.range(num_steps, dtype=tf.int64)[None, :] <
                   episode_lengths[:, None])
    return steps, valid_steps, episode_lengths

  @tf.Module.with_name_scope
  def get_episode(self, batch_size: Optional[int] = None,
                  truncate_episode_at: Optional[int] = None) -> Tuple[
                      EnvStep, Union[np.ndarray, tf.Tensor]]:
    batch_size_ = batch_size
    if batch_size_ is None:
      batch_size_ = 1

    if self._last_episode_id < 0:
      raise ValueError('No episodes in the dataset.')

    sampled_episode_ids = tf.random.uniform(
        [batch_size_], minval=0, maxval = self._last_episode_id + 1,
        dtype=tf.int64)
    steps, valid_steps = self._get_episodes(sampled_episode_ids,
                                            truncate_episode_at)

    if batch_size is None:
      steps = tf.nest.map_structure(lambda t: tf.squeeze(t, 0), steps)
      valid_steps = tf.squeeze(valid_steps, 0)

    return steps, valid_steps

  @tf.Module.with_name_scope
  def get_all_steps(self, num_steps: Optional[int] = None,
                    limit: Optional[int] = None) -> EnvStep:
    if self.num_steps <= 0:
      raise ValueError('No steps in the dataset.')

    num_steps_ = 1
    if num_steps is not None:
      num_steps_ = num_steps

    max_range = self._last_valid_steps_id + 1
    if limit is not None:
      max_range = tf.minimum(max_range, tf.cast(limit, tf.int64))
    all_valid_steps = self._valid_steps_table.read(tf.range(max_range))

    # Can't collect trajectories that trail off end of dataset.
    if tf.reduce_min(all_valid_steps) + num_steps_ > self._last_step_id + 1:
      raise ValueError('Not enough steps in the dataset.')
    all_valid_steps = tf.gather(
        all_valid_steps,
        tf.where(all_valid_steps + num_steps_ <= self._last_step_id + 1)[:, 0])

    rows_to_get = (all_valid_steps[:, None] +
                   tf.range(num_steps_, dtype=tf.int64)[None, :])
    rows_to_get = tf.math.mod(rows_to_get, self._last_step_id + 1)
    steps = self._data_table.read(rows_to_get)
    self._last_rows_read = rows_to_get

    if num_steps is None:
      steps = tf.nest.map_structure(lambda t: tf.squeeze(t, 1), steps)
      self._last_rows_read = tf.squeeze(self._last_rows_read, 1)

    return steps

  @tf.Module.with_name_scope
  def get_all_episodes(self, truncate_episode_at: Optional[int] = None,
                       limit: Optional[int] = None) -> Tuple[
                           EnvStep, Union[np.ndarray, tf.Tensor]]:
    if self._last_episode_id < 0:
      raise ValueError('No episodes in the dataset.')

    max_range = self._last_episode_id + 1
    if limit is not None:
      max_range = tf.minimum(max_range, tf.cast(limit, tf.int64))
    episode_ids = tf.range(max_range)
    return self._get_episodes(episode_ids, truncate_episode_at)

  def get_all_episode_rewards(self, truncate_episode_at: Optional[int] = None,
                              gamma: float = 0.99) -> List[float]:

    def calculate_discounted_rewards(steps, valid_steps, gamma=0.99):
      rewards = steps.reward.numpy()
      discounted_rewards = []
      for episode_rewards, valid in zip(rewards, valid_steps):
        episode_discounted_reward = 0.0
        discount_factor = 1.0
        for reward, is_valid in zip(episode_rewards, valid):
          if not is_valid:
            break
          episode_discounted_reward += reward * discount_factor
          discount_factor *= gamma
        discounted_rewards.append(episode_discounted_reward)
      return discounted_rewards


    if self._last_episode_id < 0:
      raise ValueError('No episodes in the dataset.')

    max_range = self._last_episode_id + 1
    episode_ids = tf.range(max_range)
    steps, valid_steps, = self._get_episodes(episode_ids, truncate_episode_at)

    discounted_rewards = calculate_discounted_rewards(steps, valid_steps, gamma)
    return discounted_rewards

  def save_off(self, directory):
    """Saves the dataset, including env_step_spec and capacity."""
    if not os.path.exists(directory):
      os.makedirs(directory)

    # Save env_step_spec and capacity to a JSON file
    attributes = {
      'env_step_spec': self.serialize_env_step_spec(),  # Serialize the env_step_spec
      'capacity': self.env_capacity
    }
    with open(os.path.join(directory, 'attributes.json'), 'w') as f:
      json.dump(attributes, f)

    # Save the rest of the dataset as a TensorFlow checkpoint
    checkpoint = tf.train.Checkpoint(dataset=self)
    checkpoint_path = os.path.join(directory, 'ckpt')
    checkpoint.save(checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path} and attributes saved to {directory}/attributes.json")

  @classmethod
  def load_off(cls, directory):
    """Loads the dataset by first loading env_step_spec and capacity."""
    attributes_file = os.path.join(directory, 'attributes.json')

    # Load env_step_spec and capacity from the JSON file
    if not os.path.exists(attributes_file):
      raise ValueError(f"No attributes file found in {directory}")

    with open(attributes_file, 'r') as f:
      attributes = json.load(f)

    env_step_spec = cls.deserialize_env_step_spec(attributes['env_step_spec'])
    capacity = attributes['capacity']

    # Create a new dataset instance with loaded attributes
    dataset = cls(env_step_spec, capacity)

    # Load the checkpoint to restore variables
    checkpoint_path = tf.train.latest_checkpoint(directory)
    if not checkpoint_path:
      raise ValueError(f'No suitable checkpoint found in {directory}.')

    checkpoint = tf.train.Checkpoint(dataset=dataset)
    checkpoint.restore(checkpoint_path).expect_partial()

    print(f"Checkpoint and attributes loaded from {directory}")
    return dataset

  def serialize_env_step_spec(self):
    """Convert env_step_spec to a serializable dictionary."""

    def spec_to_dict(spec):
      return {
        'shape': spec.shape.as_list(),
        'dtype': spec.dtype.name,  # Store dtype as string (e.g., "int32")
        'name': spec.name
      }

    return {
      'step_type': spec_to_dict(self.env_step_spec.step_type),
      'step_num': spec_to_dict(self.env_step_spec.step_num),
      'observation': spec_to_dict(self.env_step_spec.observation),
      'action': spec_to_dict(self.env_step_spec.action),
      'reward': spec_to_dict(self.env_step_spec.reward),
      'discount': spec_to_dict(self.env_step_spec.discount),
      'policy_info': spec_to_dict(self.env_step_spec.policy_info),
      'env_info': spec_to_dict(self.env_step_spec.env_info),
      'other_info': spec_to_dict(self.env_step_spec.other_info)
    }

  @classmethod
  def deserialize_env_step_spec(cls, env_step_spec_dict):
    """Convert dictionary back to env_step_spec using TensorSpecs."""

    def dict_to_spec(d):
      return tf.TensorSpec(shape=d['shape'], dtype=tf.dtypes.as_dtype(d['dtype']), name=d['name'])

    return EnvStep(
      step_type=dict_to_spec(env_step_spec_dict['step_type']),
      step_num=dict_to_spec(env_step_spec_dict['step_num']),
      observation=dict_to_spec(env_step_spec_dict['observation']),
      action=dict_to_spec(env_step_spec_dict['action']),
      reward=dict_to_spec(env_step_spec_dict['reward']),
      discount=dict_to_spec(env_step_spec_dict['discount']),
      policy_info=dict_to_spec(env_step_spec_dict['policy_info']),
      env_info=dict_to_spec(env_step_spec_dict['env_info']),
      other_info=dict_to_spec(env_step_spec_dict['other_info'])
    )
