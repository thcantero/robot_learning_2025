import os
import sys
import time
import pickle
import numpy as np
import torch
import gym

import matplotlib
matplotlib.use('Agg')  # set up matplotlib backend before importing pyplot
import matplotlib.pyplot as plt

# The only wrapper available in gym 0.17.3 for video recording
from gym.wrappers import Monitor, FrameStack
from gym.wrappers.frame_stack import LazyFrames


from collections import OrderedDict
from hw8.roble.infrastructure import pytorch_util as ptu
from hw8.roble.infrastructure.atari_wrappers import ReturnWrapper, wrap_deepmind
from hw8.roble.infrastructure import utils
from hw8.roble.infrastructure.logger import Logger
from hw8.roble.agents.explore_or_exploit_agent import ExplorationOrExploitationAgent
from hw8.roble.infrastructure.dqn_utils import (
    get_wrapper_by_name,
    register_custom_envs,
)

import hw8.roble.envs

MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40

import numpy as np
import gym

class FixFrameStackDim(gym.ObservationWrapper):
    """
    Reorders environment observations from (4, 84,84,1) -> (84,84,4).
    That way, 4 channels appear in the last dimension, as expected by PreprocessAtari.
    """
    def observation(self, obs):
        # If it's a LazyFrames, convert to a NumPy array
        if isinstance(obs, LazyFrames):
            obs = np.array(obs, copy=False)

        if obs.ndim == 4 and obs.shape[0] == 4 and obs.shape[3] == 1:
            tmp = np.transpose(obs, (1,2,0,3))
            tmp = tmp[..., 0]  # shape (84,84,4)
            return tmp
        else:
            return obs

class RL_Trainer(object):
    def __init__(self, params, agent_class=None):
        self.params = params
        self.logger = Logger(self.params['logging']['logdir'])

        # Register any custom environments you have
        register_custom_envs()

        # If MsPacman-v0 doesn't exist, you can try MsPacman-v4
        env_name = self.params['env']['env_name']
        if env_name == "MsPacman-v0":
            env_name = "MsPacman-v4"

        # Create gym environment
        self.env = gym.make(env_name)
        self.eval_env = gym.make(env_name)

        # Set up video logging
        env_name = self.params['env']['env_name']
        if 'MsPacman' in env_name or 'Atari' in env_name:
            video_path = os.path.join(self.params['logging']['logdir'], "videos")
            os.makedirs(video_path, exist_ok=True)
            self.env = Monitor(self.env, video_path, force=True)
            self.eval_env = Monitor(self.eval_env, video_path, force=True)
        else:
            # Skip video logging for non-Atari environments
            pass

        # ReturnWrapper
        self.env = ReturnWrapper(self.env)
        self.eval_env = ReturnWrapper(self.eval_env)

        env_name = self.params['env']['env_name']
        if 'MsPacman' in env_name or 'Atari' in env_name:
            # Apply Atari-specific wrappers
            self.env = wrap_deepmind(self.env)
            self.env = FrameStack(self.env, 4)
            self.env = FixFrameStackDim(self.env)

            self.eval_env = wrap_deepmind(self.eval_env)
            self.eval_env = FrameStack(self.eval_env, 4)
            self.eval_env = FixFrameStackDim(self.eval_env)
        else:
            pass


        # Set random seeds
        seed = self.params['logging']['random_seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['alg']['no_gpu'],
            gpu_id=self.params['alg']['which_gpu']
        )
        self.env.seed(seed)
        self.eval_env.seed(seed)

        # Max episode length
        self.params['env']['max_episode_length'] = (
            self.params['env']['max_episode_length'] or self.env.spec.max_episode_steps
        )
        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.params['env']['max_episode_length']

        # Discrete or continuous
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        img = len(self.env.observation_space.shape) > 2
        self.params['alg']['discrete'] = discrete
        self.params['alg']['ob_dim'] = (
            self.env.observation_space.shape if img
            else self.env.observation_space.shape[0]
        )
        self.params['alg']['ac_dim'] = (
            self.env.action_space.n if discrete
            else self.env.action_space.shape[0]
        )

        self.agent = agent_class(self.env, self.params)

    def run_training_loop(self, n_iter, collect_policy, eval_policy):
        """
        Main training loop
        """
        self.total_envsteps = 0
        self.start_time = time.time()

        for itr in range(n_iter):
            print(f"\n\n********** Iteration {itr} ************")

             # If DQN skip sample_trajectories 
            if self.params['alg']['rl_alg'] == 'dqn':
                envsteps_this_batch = 0

                for _ in range(self.params['alg']['batch_size']):
                    self.agent.step_env()  # storing frames & stacking
                    envsteps_this_batch += 1.
                paths = []
            else:
            #Non-DQN
                paths, envsteps_this_batch, _ = self.collect_training_trajectories(
                    itr, collect_policy, self.params['alg']['batch_size']
                )

            self.total_envsteps += envsteps_this_batch

            # Train agent
            print("\nTraining agent...")
            all_logs = self.train_agent()

            print("\nPerforming logging...")
            self.perform_logging(itr, paths, eval_policy, all_logs)

    def collect_training_trajectories(self, itr, collect_policy, batch_size):
        """
        Collect trajectories for training
        """
        print(f"Collecting {batch_size} transitions using current policy...")
        paths, envsteps_this_batch = utils.sample_trajectories(
            self.env,
            collect_policy,
            batch_size,
            self.params['env']['max_episode_length']
        )
        return paths, envsteps_this_batch, None

    def train_agent(self):
        """
        Train the agent using the replay buffer
        """
        print("\nTraining agent from replay buffer data...")
        all_logs = []
        for _ in range(self.params['alg']['num_agent_train_steps_per_iter']):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(
                self.params['alg']['train_batch_size']
            )
            train_log = self.agent.train(
                ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch
            )
            all_logs.append(train_log)
        return all_logs

    def perform_logging(self, itr, paths, eval_policy, all_logs):
        """
        Evaluate & log results
        """
        last_log = all_logs[-1]

        # Evaluate
        eval_paths, _ = utils.sample_trajectories(
            self.eval_env,
            eval_policy,
            self.params['alg']['eval_batch_size'],
            self.params['env']['max_episode_length']
        )

        # If 'paths' is empty (typical for DQN), we avoid NaN
        if len(paths) == 0:
            train_returns = []
            train_avg_return = 0.0  # or you could do None
        else:
            train_returns = [p["reward"].sum() for p in paths]
            train_avg_return = np.mean(train_returns)

        eval_returns = [p["reward"].sum() for p in eval_paths]
        eval_avg_return = np.mean(eval_returns) if len(eval_returns) else 0.0

        import collections
        logs = collections.OrderedDict()
        logs["Train_AverageReturn"] = train_avg_return
        logs["Eval_AverageReturn"] = eval_avg_return

        # Merge last train logs
        logs.update(last_log)

        logs["TotalEnvSteps"] = self.total_envsteps

        for key, value in logs.items():
            print(f"{key} : {value}")
            self.logger.log_scalar(value, key, itr)

        print(f"Writing logs to CSV at: {self.logger.log_file_path}")    
        self.logger.log_file(itr, logs)

        self.logger.flush()
