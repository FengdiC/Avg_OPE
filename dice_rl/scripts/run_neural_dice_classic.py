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
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
granddir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, granddir)
import tensorflow.compat.v2 as tf
tf.compat.v1.enable_v2_behavior()
import pickle, csv
from tqdm import tqdm

os.environ["TF_USE_LEGACY_KERAS"]='1'

from dice_rl.estimators.neural_dice import NeuralDice
from dice_rl.estimators import estimator as estimator_lib
from dice_rl.networks.value_network import ValueNetwork
import dice_rl.utils.common as common_utils
from dice_rl.data.dataset import Dataset, EnvStep, StepType
from dice_rl.data.tf_offpolicy_dataset import TFOffpolicyDataset
import torch
import torch.distributions as td
import dice_rl.ppo.algo.core as core
import gymnasium as gym
from dice_rl.scripts.create_dataset_pytorch import create_env_step_spec_from_gym, PPOBuffer

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', None, 'Directory to load dataset from.')
flags.DEFINE_string('path', None, 'Directory to load dataset from.')
flags.DEFINE_string('output_dir', None,
                    'Directory to save the model and estimation results.')
flags.DEFINE_string('env_name', 'grid', 'Environment name.')
flags.DEFINE_integer('seed', 0, 'Initial random seed.')
flags.DEFINE_bool('tabular_obs', False, 'Whether to use tabular observations.')

flags.DEFINE_integer('num_trajectory', 1000,
                     'Number of trajectories to collect.')
flags.DEFINE_integer('max_trajectory_length', 40,
                     'Cutoff trajectory at this step.')
flags.DEFINE_float('alpha', 0.0, 'How close to target policy.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor.')
flags.DEFINE_float('array', 0, 'slurm array number')

flags.DEFINE_float('nu_learning_rate', 0.0001, 'Learning rate for nu.')
flags.DEFINE_float('zeta_learning_rate', 0.0001, 'Learning rate for zeta.')
flags.DEFINE_float('nu_regularizer', 0.0, 'Ortho regularization on nu.')
flags.DEFINE_float('zeta_regularizer', 0.0, 'Ortho regularization on zeta.')
flags.DEFINE_integer('epoch', 100000, 'Number of training steps.')
flags.DEFINE_integer('steps', 5, 'Number of training steps.')

flags.DEFINE_integer('batch_size', 512, 'Batch size.')

flags.DEFINE_float('f_exponent', 2, 'Exponent for f function.')
flags.DEFINE_bool('primal_form', False,
                  'Whether to use primal form of loss for nu.')

flags.DEFINE_float('primal_regularizer', 0.,
                   'LP regularizer of primal variables.')
flags.DEFINE_float('dual_regularizer', 1., 'LP regularizer of dual variables.')
flags.DEFINE_bool('zero_reward', False,
                  'Whether to ignore reward in optimization.')
flags.DEFINE_float('norm_regularizer', 1.,
                   'Weight of normalization constraint.')
flags.DEFINE_bool('zeta_pos', True, 'Whether to enforce positivity constraint.')

flags.DEFINE_float('scale_reward', 1., 'Reward scaling factor.')
flags.DEFINE_float('shift_reward', 0., 'Reward shift factor.')
flags.DEFINE_string(
    'transform_reward', None, 'Non-linear reward transformation'
    'One of [exp, cuberoot, None]')
flags.DEFINE_float('random_weight', 0.2,'random policy')


class ActionDistributionWrapper:
    """Wrapper for PyTorch action distribution, providing a compatible API."""

    def __init__(self, distribution):
        self.distribution = distribution  # Store the PyTorch distribution
        self.action = distribution  # Store the sampled action

    def probs_parameter(self):
        """Returns the probabilities for categorical actions (or mean for continuous actions)."""
        if hasattr(self.action, 'probs'):
            return self.action.probs
        else:
            raise AttributeError("Action doesn't have `probs` or `mean` attributes.")


class PyTorchPolicyWrapper:
    """A wrapper to use PyTorch policy in TensorFlow environments."""

    def __init__(self, torch_policy, mujoco=False, random_weight=0.2):
        self.torch_policy = torch_policy  # PyTorch actor-critic policy
        self.mujoco = mujoco
        self.random_weight = random_weight

    def distribution(self, tf_time_step):
        """Generates action distribution for a given time step (compatible with TensorFlow)."""
        # Convert TensorFlow observation to a NumPy array and then a PyTorch tensor
        obs = tf_time_step.observation.numpy()  # Run in eager mode
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)

        # Get action distribution from the PyTorch policy
        pi = self.torch_policy.pi._distribution(obs_tensor)

        if self.mujoco:
            std = torch.exp(self.torch_policy.pi.log_std)
            mu = self.torch_policy.pi.mu_net(obs_tensor)
            action_weights = torch.distributions.Normal(mu, std * self.random_weight)
        else:
            action_weights = pi.probs  # Assuming `pi` has a `probs` or similar method

        # Convert the PyTorch result to TensorFlow tensor
        # print("action_weights", action_weights)
        action_weights_tf = tf.convert_to_tensor(action_weights.detach().numpy(), dtype=tf.float32)

        return action_weights_tf

    def action(self, tf_time_step):
        """Sample actions from the PyTorch policy."""
        obs = tf_time_step.observation.numpy()
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)

        # Sample action based on the PyTorch policy distribution
        if self.mujoco:
            std = torch.exp(self.torch_policy.pi.log_std)
            mu = self.torch_policy.pi.mu_net(obs_tensor)
            pi = torch.distributions.Normal(mu, std )
            action = pi.sample().detach().numpy()
        else:
            targ_a, _, _ = self.torch_policy.step(obs_tensor)
            action = targ_a.detach().numpy()
        return tf.convert_to_tensor(action, dtype=tf.float32)

def load_dataset(log_dir,name,buffer_size,max_len,env_name,action_discrete=True):
    env_step_spec = create_env_step_spec_from_gym(env_name)
    tf_dataset = TFOffpolicyDataset(
        env_step_spec,
        capacity=buffer_size+1)
    with open(log_dir + name + '.pkl', 'rb') as outp:
        buf = pickle.load(outp)
    i = 0
    while i < buffer_size:
        step_type = StepType.FIRST if i % max_len == 0 else StepType.MID
        if i % max_len == max_len - 1:
            step_type = StepType.LAST

        if not action_discrete:
            tf_action = tf.convert_to_tensor(buf.act_buf[i], dtype=tf.float32)
        else:
            tf_action = tf.convert_to_tensor(int(buf.act_buf[i]), dtype=tf.int32)

        # Create the EnvStep using cached step data
        env_step = EnvStep(
            step_type=tf.convert_to_tensor(step_type, dtype=tf.int32),
            step_num=tf.convert_to_tensor(buf.tim_buf[i], dtype=tf.int64),
            observation=tf.convert_to_tensor(buf.obs_buf[i], dtype=tf.float32),
            action=tf_action,
            reward=tf.convert_to_tensor(buf.rew_buf[i], dtype=tf.float32),
            discount=tf.convert_to_tensor(buf.gamma, dtype=tf.float32),
            # should be behaviour policy
            policy_info={'log_probability':tf.convert_to_tensor(buf.logbev_buf[i], dtype=tf.float32)},
            env_info=tf.convert_to_tensor(0, dtype=tf.int32),
            other_info=tf.convert_to_tensor(0, dtype=tf.int32)
        )
        tf_dataset.add_step(env_step)
        i+=1
    return tf_dataset

# load target policy
def load(path,env):
    ac_kwargs = dict(hidden_sizes=[64,32])

    ac = core.MLPActorCritic(env.observation_space, env.action_space, **ac_kwargs)
    checkpoint = torch.load(path)
    ac.load_state_dict(checkpoint['model_state_dict'])
    return ac

def main(argv):
    seeds = range(10)
    tf.config.run_functions_eagerly(True)

    if FLAGS.array > 200:
        return -1

    discount_factor_lists = [0.8, 0.9, 0.95, 0.99, 0.995]
    size_lists = [2000, 4000, 8000, 16000]

    weight_lists = [0.1, 0.2, 0.3, 0.4, 0.5]
    length_lists = [20, 40, 80, 100]
    env = ['CartPole-v1', 'Acrobot-v1']
    path = ['./exper/cartpole.pth', './exper/acrobot.pth']
    idx = np.unravel_index(int(FLAGS.array), (5, 4, 5, 2))
    random_weight, max_trajectory_length, gamma  = (
        weight_lists[idx[0]],
        length_lists[idx[1]],
        discount_factor_lists[idx[2]],
    )
    env_name, path = env[idx[3]], path[idx[3]]

    nu_learning_rate = FLAGS.nu_learning_rate
    zeta_learning_rate = FLAGS.zeta_learning_rate
    nu_regularizer = FLAGS.nu_regularizer
    zeta_regularizer = FLAGS.zeta_regularizer
    batch_size = FLAGS.batch_size

    f_exponent = FLAGS.f_exponent
    primal_form = FLAGS.primal_form

    primal_regularizer = FLAGS.primal_regularizer
    dual_regularizer = FLAGS.dual_regularizer
    zero_reward = FLAGS.zero_reward
    norm_regularizer = FLAGS.norm_regularizer
    zeta_pos = FLAGS.zeta_pos

    scale_reward = FLAGS.scale_reward
    shift_reward = FLAGS.shift_reward
    transform_reward = FLAGS.transform_reward

    def reward_fn(env_step):
        reward = env_step.reward * scale_reward + shift_reward
        if transform_reward is None:
          return reward
        if transform_reward == 'exp':
          reward = tf.math.exp(reward)
        elif transform_reward == 'cuberoot':
          reward = tf.sign(reward) * tf.math.pow(tf.abs(reward), 1.0 / 3.0)
        else:
          raise ValueError('Reward {} not implemented.'.format(transform_reward))
        return reward

    with open(FLAGS.data_dir+'/classic_obj.pkl','rb') as file:
        obj = pickle.load(file)
    true_obj = obj[env_name][idx[2]]

    os.makedirs(FLAGS.output_dir, exist_ok=True)
    os.makedirs(FLAGS.output_dir+str(env_name), exist_ok=True)
    filename = FLAGS.output_dir + str(env_name)+'/dice-classic-' + str(env_name) + '-discount-' + str(gamma) \
               + '-length-' + str(max_trajectory_length) + '-random-' + str(random_weight) + '.csv'

    mylist = [str(i) for i in range(0, FLAGS.epoch * FLAGS.steps, FLAGS.steps)] + ['hyperparam']
    with open(filename, 'w+', newline='') as file:
        # Step 4: Using csv.writer to write the list to the CSV file
        writer = csv.writer(file)
        writer.writerow(mylist)  # Use writerow for single list

    for seed in tqdm(seeds, desc="Seeds"):
        # Set seeds
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        tf.random.set_seed(seed)
        for size in size_lists:
            name = ['discount_factor', 0.8, 'random_weight', random_weight, 'max_length', max_trajectory_length,
                    'buffer_size', 16000, 'seed', seed, 'env', env_name]
            name = '-'.join(str(x) for x in name)

            dataset = load_dataset(FLAGS.data_dir,'/dataset/'+ name, size,max_trajectory_length,env_name,
                                   action_discrete=True)

            name = ['discount_factor', 0.8, 'random_weight', random_weight, 'max_length', max_trajectory_length,
                    'buffer_size', 16000, 'seed', seed + 1314, 'env', env_name]
            name = '-'.join(str(x) for x in name)

            dataset2 = load_dataset(FLAGS.data_dir,'/dataset_test/'+name, size, max_trajectory_length, env_name,
                                   action_discrete=True)

            all_steps = dataset.get_all_steps()
            max_reward = tf.reduce_max(all_steps.reward)
            min_reward = tf.reduce_min(all_steps.reward)
            print('num loaded steps', dataset.num_steps)
            print('num loaded total steps', dataset.num_total_steps)
            print('num loaded episodes', dataset.num_episodes)
            print('num loaded total episodes', dataset.num_total_episodes)
            print('min reward', min_reward, 'max reward', max_reward)
            print('behavior per-step',
                  estimator_lib.get_fullbatch_average(dataset, gamma=gamma))

            activation_fn = tf.nn.relu
            kernel_initializer = tf.keras.initializers.GlorotUniform()
            hidden_dims = (256,256)
            input_spec = (dataset.spec.observation, dataset.spec.action)
            print(dataset.spec.observation, dataset.spec.action)
            nu_network = ValueNetwork(
                input_spec,
                fc_layer_params=hidden_dims,
                activation_fn=activation_fn,
                kernel_initializer=kernel_initializer,
                last_kernel_initializer=kernel_initializer)
            output_activation_fn = tf.math.square if zeta_pos else tf.identity
            zeta_network = ValueNetwork(
                input_spec,
                fc_layer_params=hidden_dims,
                activation_fn=activation_fn,
                output_activation_fn=output_activation_fn,
                kernel_initializer=kernel_initializer,
                last_kernel_initializer=kernel_initializer)

            nu_optimizer = tf.keras.optimizers.Adam(nu_learning_rate, clipvalue=1.0)
            zeta_optimizer = tf.keras.optimizers.Adam(zeta_learning_rate, clipvalue=1.0)
            lam_optimizer = tf.keras.optimizers.Adam(nu_learning_rate, clipvalue=1.0)

            estimator = NeuralDice(
                dataset.spec,
                nu_network,
                zeta_network,
                nu_optimizer,
                zeta_optimizer,
                lam_optimizer,
                gamma,
                zero_reward=zero_reward,
                f_exponent=f_exponent,
                primal_form=primal_form,
                reward_fn=reward_fn,
                primal_regularizer=primal_regularizer,
                dual_regularizer=dual_regularizer,
                norm_regularizer=norm_regularizer,
                nu_regularizer=nu_regularizer,
                zeta_regularizer=zeta_regularizer)

            global_step = tf.Variable(0, dtype=tf.int64)
            tf.summary.experimental.set_step(global_step)

            env = gym.make(env_name)
            env.reset(seed=seed)
            env.action_space.seed(seed)
            ac = load(path, env)
            target_policy = PyTorchPolicyWrapper(ac,mujoco=False,random_weight=random_weight)
            running_losses = []
            running_estimates = []
            train_estimates = []
            test_estimates = []

            num_steps = FLAGS.epoch * FLAGS.steps
            best=5
            name = ['discount_factor', gamma, 'random_weight', random_weight, 'max_length', max_trajectory_length,
                    'env', env_name, 'buffer_size', size, 'seed', seed]
            name = '-'.join(str(x) for x in name)
            for step in range(num_steps):
                transitions_batch = dataset.get_step(batch_size, num_steps=2)
                initial_steps_batch, _ = dataset.get_episode(batch_size, truncate_episode_at=1)
                initial_steps_batch = tf.nest.map_structure(lambda t: t[:, 0, ...],
                                                            initial_steps_batch)
                losses = estimator.train_step(initial_steps_batch, transitions_batch,
                                              target_policy)
                running_losses.append(losses)

                if step % FLAGS.steps == 0 or step == num_steps - 1:
                    eval_obj = estimator.eval_policy_csv(dataset, target_policy)
                    eval_obj2 = estimator.eval_policy_csv(dataset2, target_policy)
                    train_estimates.append(eval_obj)
                    test_estimates.append(eval_obj2)
                    # if (eval_obj2-true_obj)**2 < best:
                    #     best = (eval_obj2-true_obj)**2
                    #     # Save the model at the end of training
                    #     if FLAGS.output_dir is not None:
                    #         model_save_path = os.path.join(FLAGS.output_dir, env_name,name+'_best_model_weights')
                    #
                    #         # Create a checkpoint object
                    #         checkpoint = tf.train.Checkpoint(nu_network=nu_network,
                    #                                          zeta_network=zeta_network,
                    #                                          nu_optimizer=nu_optimizer,
                    #                                          zeta_optimizer=zeta_optimizer,
                    #                                          lam_optimizer=lam_optimizer)
                    #
                    #         # Save the checkpoint
                    #         checkpoint.save(os.path.join(model_save_path, 'checkpoint'))
                    #
                    #         print(f"Model saved at {model_save_path}")


                # if (step < 1000 and step % 25 == 0) or (step >= 1000 and step % 100 == 0):
                #     plot_file = os.path.join(output_dir, '_zeta_'+str(step))
                #     estimator.plot_zeta_csv(dataset, target_policy, filename_prefix=plot_file)

                global_step.assign_add(1)

            # Save the model at the end of training
            if FLAGS.output_dir is not None:
                model_save_path = os.path.join(FLAGS.output_dir, env_name,name+'_last_model_weights')

                # Create a checkpoint object
                checkpoint = tf.train.Checkpoint(nu_network=nu_network,
                                                 zeta_network=zeta_network,
                                                 nu_optimizer=nu_optimizer,
                                                 zeta_optimizer=zeta_optimizer,
                                                 lam_optimizer=lam_optimizer)

                # Save the checkpoint
                checkpoint.save(os.path.join(model_save_path, 'checkpoint'))

                print(f"Model saved at {model_save_path}")

            train, test = np.around(train_estimates, decimals=4), np.around(test_estimates, decimals=4)
            mylist = [str(i) for i in list(train)] + ['-'.join(['train', 'size', str(size), 'seed', str(seed)])]
            with open(filename, 'a', newline='') as file:
                # Step 4: Using csv.writer to write the list to the CSV file
                writer = csv.writer(file)
                writer.writerow(mylist)  # Use writerow for single list
            mylist = [str(i) for i in list(test)] + ['-'.join(['test', 'size', str(size), 'seed', str(seed)])]
            with open(filename, 'a', newline='') as file:
                # Step 4: Using csv.writer to write the list to the CSV file
                writer = csv.writer(file)
                writer.writerow(mylist)  # Use writerow for single list

    print('Done!')


if __name__ == '__main__':
  app.run(main)
